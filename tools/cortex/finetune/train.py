__copyright__ = "Copyright (c) 2026 Alex Laird"
__license__ = "MIT"

import unsloth  # noqa: F401 — must be imported before trl/transformers/peft

import argparse
import gc
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from run_helper import banner, follow
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

PRETRAIN_ADAPTER_DIRNAME = "lora-pretrain"


def _clear_memory():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception as exc:  # pragma: no cover - best effort only
        logger.debug(f"Skipping CUDA cleanup: {exc}")


def _save_merged_model(model, tokenizer, merged_path):
    logger.info(f"Merging adapter into full weights at {merged_path} ...")
    model.save_pretrained_merged(str(merged_path), tokenizer, save_method="merged_16bit")
    logger.info("Merge complete.")


def _has_saved_artifacts(path: Path | None) -> bool:
    return bool(path and path.exists() and path.is_dir() and any(path.iterdir()))


def resolve_pretrain_adapter_path(output_path, pretrain_adapter_path=None, base_only=False):
    if base_only:
        return None

    if pretrain_adapter_path is not None:
        if not _has_saved_artifacts(pretrain_adapter_path):
            raise FileNotFoundError(
                f"Requested pre-train adapter does not exist or is empty: {pretrain_adapter_path}"
            )
        return pretrain_adapter_path

    candidate = output_path / PRETRAIN_ADAPTER_DIRNAME
    if _has_saved_artifacts(candidate):
        return candidate

    raise FileNotFoundError(
        f"Missing required pre-train adapter: {candidate}. "
        "Run pretrain_tone.py first or pass --base-only to skip tone pre-training."
    )


def load_model_and_tokenizer(pretrain_adapter_path=None):
    from unsloth import FastLanguageModel

    model_name = str(pretrain_adapter_path) if pretrain_adapter_path else HF_MODEL_ID
    logger.info(
        f"Loading {'pre-train adapter' if pretrain_adapter_path else 'base model'}: {model_name}"
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    if not pretrain_adapter_path:
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


def train(data_path, output_path, resume, pretrain_adapter_path=None, base_only=False):
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    from unsloth.chat_templates import get_chat_template, standardize_sharegpt

    resolved_pretrain_adapter = resolve_pretrain_adapter_path(
        output_path,
        pretrain_adapter_path=pretrain_adapter_path,
        base_only=base_only,
    )
    model, tokenizer = load_model_and_tokenizer(pretrain_adapter_path=resolved_pretrain_adapter)
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
    return model, tokenizer


def merge(output_path):
    from unsloth import FastLanguageModel

    adapter_path = output_path / "lora-adapter"
    merged_path = output_path / "merged"

    _clear_memory()

    model = None
    tokenizer = None
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_path),
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        _save_merged_model(model, tokenizer, merged_path)
    finally:
        del model
        del tokenizer
        _clear_memory()


def train_and_merge(data_path, output_path, resume, pretrain_adapter_path=None, base_only=False):
    model, tokenizer = train(
        data_path,
        output_path,
        resume,
        pretrain_adapter_path=pretrain_adapter_path,
        base_only=base_only,
    )
    try:
        _clear_memory()
        _save_merged_model(model, tokenizer, output_path / "merged")
    finally:
        del model
        del tokenizer
        _clear_memory()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4 with QLoRA via Unsloth.")
    parser.add_argument("--data", type=Path, default=None, help="Path to sharegpt.jsonl")
    parser.add_argument("--output", type=Path, default=None, help="Output directory for adapter")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    parser.add_argument(
        "--pretrain-adapter",
        type=Path,
        default=None,
        help="LoRA adapter from pretrain_tone.py. Defaults to OUTPUT/lora-pretrain.",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Ignore any pre-train adapter and fine-tune directly from the base model.",
    )
    parser.add_argument("--merge-only", action="store_true", help="Skip training; merge existing adapter")
    parser.add_argument("--bg", action="store_true", help="Run in background after prompts (internal use)")
    args = parser.parse_args()

    data_path = args.data or (FINETUNE_DATA_DIR / "sharegpt.jsonl")
    output_path = args.output or FINETUNE_OUTPUT_DIR

    if not args.bg:
        log_path = Path("logs") / "train.log"
        log_path.parent.mkdir(exist_ok=True)
        cmd = [
            sys.executable, __file__,
            "--data", str(data_path),
            "--output", str(output_path),
            "--bg",
        ]
        if args.resume:
            cmd += ["--resume"]
        if args.pretrain_adapter is not None:
            cmd += ["--pretrain-adapter", str(args.pretrain_adapter)]
        if args.base_only:
            cmd += ["--base-only"]
        if args.merge_only:
            cmd += ["--merge-only"]
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, start_new_session=True)
        follow(proc, log_path, "train")
    elif args.merge_only:
        banner("MERGE — STARTING")
        merge(output_path)
        banner("MERGE — DONE")
    else:
        banner("TRAIN — STARTING")
        train_and_merge(
            data_path,
            output_path,
            args.resume,
            pretrain_adapter_path=args.pretrain_adapter,
            base_only=args.base_only,
        )
        banner("TRAIN — DONE")
