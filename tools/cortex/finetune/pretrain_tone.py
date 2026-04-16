__copyright__ = "Copyright (c) 2026 Alex Laird"
__license__ = "MIT"

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from config import (
    BLOG_OUTPUT_DIR,
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


def pretrain(blog_dir, output_path, resume):
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    txt_files = sorted(blog_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {blog_dir} — run fetch-blog first")

    logger.info(f"Loading model: {HF_MODEL_ID}")
    model, tokenizer = load_model_and_tokenizer()

    logger.info(f"Loading {len(txt_files)} blog posts from {blog_dir}")
    texts = [f.read_text(encoding="utf-8").strip() for f in txt_files]
    dataset = Dataset.from_dict({"text": texts})

    adapter_path = output_path / "lora-pretrain"
    adapter_path.mkdir(parents=True, exist_ok=True)

    resume_from = str(adapter_path) if resume and any(adapter_path.iterdir()) else None

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
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

    logger.info("Starting tone pre-training ...")
    trainer.train(resume_from_checkpoint=resume_from)

    logger.info(f"Saving pre-train LoRA adapter to {adapter_path}")
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info("Pre-training complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Continued pre-training on blog posts for tone absorption.")
    parser.add_argument("--blog-dir", type=Path, default=None, help="Directory of .txt blog post files")
    parser.add_argument("--output", type=Path, default=None, help="Output directory for pre-train adapter")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    args = parser.parse_args()

    blog_dir = args.blog_dir or BLOG_OUTPUT_DIR
    output_path = args.output or FINETUNE_OUTPUT_DIR

    pretrain(blog_dir=blog_dir, output_path=output_path, resume=args.resume)
