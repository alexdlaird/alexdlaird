__copyright__ = "Copyright (c) 2026 Alex Laird"
__license__ = "MIT"

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from config import FINETUNE_OUTPUT_DIR, HF_MODEL_ID, MAX_SEQ_LENGTH, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Preferred: derive TEMPLATE + PARAMETER lines from stock `gemma4`'s Modelfile.
# Unsloth's GGUF export doesn't reliably embed `tokenizer.chat_template` in
# metadata, so without an explicit TEMPLATE Ollama falls back to a generic frame
# that doesn't match how the model was trained — the model then generates but
# never emits <end_of_turn>. Shelling out to `ollama show` keeps us in lockstep
# with upstream tuning.
STOCK_BASE_MODEL = "gemma4"
_STRIP_DIRECTIVES = {"FROM", "SYSTEM"}

# Used only when stock gemma4's Modelfile can't be read at export time.
MODELFILE_FALLBACK = """FROM __GGUF_PATH__
TEMPLATE \"\"\"<start_of_turn>user
{{ if .System }}{{ .System }}

{{ end }}{{ .Prompt }}<end_of_turn>
<start_of_turn>model
{{ .Response }}<end_of_turn>
\"\"\"
PARAMETER stop "<end_of_turn>"
PARAMETER stop "<start_of_turn>"
PARAMETER temperature 1.0
PARAMETER top_k 64
PARAMETER top_p 0.95
PARAMETER num_ctx 8192
"""


def _build_modelfile(gguf_path, system_prompt):
    result = subprocess.run(
        ["ollama", "show", "--modelfile", STOCK_BASE_MODEL],
        capture_output=True, text=True, check=False,
    )
    if result.returncode == 0 and "TEMPLATE" in result.stdout:
        logger.info(f"Deriving Modelfile from stock {STOCK_BASE_MODEL} (TEMPLATE + PARAMETER)")
        body = _rewrite_from_stock(result.stdout, gguf_path)
    else:
        logger.warning(
            f"`ollama show --modelfile {STOCK_BASE_MODEL}` unavailable — using hardcoded fallback"
        )
        body = MODELFILE_FALLBACK.replace("__GGUF_PATH__", str(gguf_path))
    return body.rstrip() + f'\n\nSYSTEM """{system_prompt}"""\n'


def _rewrite_from_stock(stock_modelfile, gguf_path):
    """Replace FROM with our GGUF path and drop any SYSTEM block; keep everything else."""
    out = [f"FROM {gguf_path}"]
    lines = stock_modelfile.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        directive = stripped.split(None, 1)[0]
        if directive in _STRIP_DIRECTIVES:
            # Skip a potentially multi-line triple-quoted value.
            if '"""' in line and line.count('"""') == 1:
                i += 1
                while i < len(lines) and '"""' not in lines[i]:
                    i += 1
            i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


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

    # Unsloth may ignore output_path and write to {adapter_name}_gguf/ alongside the adapter
    unsloth_output = adapter_path.parent / (adapter_path.name + "_gguf")

    # Clear GGUFs from prior runs so the Modelfile's FROM always points at a
    # freshly-hashed file. Without this, a stale file in one output dir can
    # shadow a new file in the other, and Ollama later sees a digest mismatch.
    for stale in list(output_path.glob("*.gguf")) + list(unsloth_output.glob("*.gguf")):
        logger.info(f"Removing stale GGUF: {stale}")
        stale.unlink()

    logger.info(f"Exporting GGUF ({quant}) to {output_path} ...")
    model.save_pretrained_gguf(str(output_path), tokenizer, quantization_method=quant)

    gguf_files = list(output_path.glob("*.gguf")) + list(unsloth_output.glob("*.gguf"))
    if not gguf_files:
        logger.error("No .gguf file found after export — check Unsloth output.")
        return

    quantized = [f for f in gguf_files if "BF16" not in f.name]
    gguf_path = (quantized[0] if quantized else gguf_files[0]).resolve()

    modelfile_path = output_path / "Modelfile"
    modelfile_path.write_text(_build_modelfile(gguf_path, SYSTEM_PROMPT))

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
