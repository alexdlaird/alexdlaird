__copyright__ = "Copyright (c) 2026 Alex Laird"
__license__ = "MIT"

import unsloth  # noqa: F401 — must be imported before trl/transformers/peft

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from config import (
    FINETUNE_DATA_DIR,
    FINETUNE_OUTPUT_DIR,
    GRADIENT_ACCUMULATION_STEPS,
    HF_MODEL_ID,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MAX_SEQ_LENGTH,
    NUM_EPOCHS,
    TRAIN_BATCH_SIZE,
    WARMUP_RATIO,
)

logger = logging.getLogger(__name__)


def load_model_and_tokenizer():
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=HF_MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model, tokenizer


def train(data_path, output_path, resume):
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    from unsloth.chat_templates import get_chat_template, standardize_sharegpt

    logger.info(f"Loading model: {HF_MODEL_ID}")
    model, tokenizer = load_model_and_tokenizer()
    tokenizer = get_chat_template(tokenizer, chat_template="gemma")

    logger.info(f"Loading training data: {data_path}")
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    dataset = standardize_sharegpt(dataset)

    def apply_template(examples):
        texts = tokenizer.apply_chat_template(examples["conversations"], tokenize=False)
        return {"text": texts}

    dataset = dataset.map(apply_template, batched=True)

    adapter_path = output_path / "lora-adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)

    resume_from = str(adapter_path) if resume and any(adapter_path.iterdir()) else None

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_num_proc=2,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_ratio=WARMUP_RATIO,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=False,
            bf16=True,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            output_dir=str(adapter_path),
            report_to="none",
        ),
    )

    logger.info("Starting training ...")
    trainer.train(resume_from_checkpoint=resume_from)

    logger.info(f"Saving LoRA adapter to {adapter_path}")
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info("Training complete.")


def merge(output_path):
    from unsloth import FastLanguageModel

    adapter_path = output_path / "lora-adapter"
    merged_path = output_path / "merged"

    logger.info(f"Merging adapter into full weights at {merged_path} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    model.save_pretrained_merged(str(merged_path), tokenizer, save_method="merged_16bit")
    logger.info("Merge complete.")


def train_and_merge(data_path, output_path, resume):
    train(data_path, output_path, resume)
    merge(output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4 with QLoRA via Unsloth.")
    parser.add_argument("--data", type=Path, default=None, help="Path to sharegpt.jsonl")
    parser.add_argument("--output", type=Path, default=None, help="Output directory for adapter")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    parser.add_argument("--merge-only", action="store_true", help="Skip training; merge existing adapter")
    parser.add_argument("--bg", action="store_true", help="Run in background after prompts (internal use)")
    args = parser.parse_args()

    data_path = args.data or (FINETUNE_DATA_DIR / "sharegpt.jsonl")
    output_path = args.output or FINETUNE_OUTPUT_DIR

    if not args.bg:
        import os
        import subprocess
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / "train.log"
        cmd = [
            sys.executable, __file__,
            "--data", str(data_path),
            "--output", str(output_path),
            "--bg",
        ]
        if args.resume:
            cmd += ["--resume"]
        if args.merge_only:
            cmd += ["--merge-only"]
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, start_new_session=True)
        print(f"\n--> train running in background (PID {proc.pid}), tailing log ...")
        os.execlp("tail", "tail", "-f", str(log_path))
    elif args.merge_only:
        merge(output_path)
    else:
        train_and_merge(data_path, output_path, args.resume)
