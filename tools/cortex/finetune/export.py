__copyright__ = "Copyright (c) 2026 Alex Laird"
__license__ = "MIT"

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from config import FINETUNE_OUTPUT_DIR, HF_MODEL_ID, MAX_SEQ_LENGTH, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

MODELFILE_TEMPLATE = """FROM {gguf_path}
PARAMETER stop "<end_of_turn>"
PARAMETER stop "<start_of_turn>"
SYSTEM \"\"\"{system_prompt}\"\"\"
"""


def export(adapter_path, output_path, quant, model_name):
    from unsloth import FastLanguageModel

    logger.info(f"Loading adapter from {adapter_path} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting GGUF ({quant}) to {output_path} ...")
    model.save_pretrained_gguf(str(output_path), tokenizer, quantization_method=quant)

    gguf_files = list(output_path.glob("*.gguf"))
    if not gguf_files:
        # Unsloth may ignore output_path and write to {adapter_name}_gguf/ alongside the adapter
        unsloth_output = adapter_path.parent / (adapter_path.name + "_gguf")
        gguf_files = list(unsloth_output.glob("*.gguf"))
    if not gguf_files:
        logger.error("No .gguf file found after export — check Unsloth output.")
        return

    quantized = [f for f in gguf_files if "BF16" not in f.name]
    gguf_path = (quantized[0] if quantized else gguf_files[0]).resolve()

    output_path.mkdir(parents=True, exist_ok=True)
    modelfile_path = output_path / "Modelfile"
    modelfile_path.write_text(
        MODELFILE_TEMPLATE.format(gguf_path=gguf_path, system_prompt=SYSTEM_PROMPT)
    )

    logger.info(f"Modelfile written to {modelfile_path}")
    logger.info("")
    logger.info("To register with Ollama, run:")
    logger.info(f"  ollama create {model_name} -f {modelfile_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Export fine-tuned model to GGUF for Ollama.")
    parser.add_argument("--adapter", type=Path, default=None, help="Path to LoRA adapter or merged weights")
    parser.add_argument("--output", type=Path, default=None, help="Output directory for GGUF and Modelfile")
    parser.add_argument("--quant", default="q4_k_m", choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
                        help="Quantization method (default: q4_k_m)")
    parser.add_argument("--model-name", default="cortex", help="Name for ollama create (default: cortex)")
    parser.add_argument("--bg", action="store_true", help="Run in background (internal use)")
    args = parser.parse_args()

    adapter_path = args.adapter or (FINETUNE_OUTPUT_DIR / "merged")
    output_path = args.output or (FINETUNE_OUTPUT_DIR / "gguf")

    if not args.bg:
        import os
        import subprocess
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / "export.log"
        cmd = [
            sys.executable, __file__,
            "--adapter", str(adapter_path),
            "--output", str(output_path),
            "--quant", args.quant,
            "--model-name", args.model_name,
            "--bg",
        ]
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, start_new_session=True)
        print(f"\n--> export running in background (PID {proc.pid}), tailing log ...")
        os.execlp("tail", "tail", "-f", str(log_path))
    else:
        export(adapter_path, output_path, args.quant, args.model_name)
