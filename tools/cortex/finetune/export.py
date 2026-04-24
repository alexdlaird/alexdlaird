__copyright__ = "Copyright (c) 2026 Alex Laird"
__license__ = "MIT"

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from config import FINETUNE_OUTPUT_DIR, HF_MODEL_ID, MAX_SEQ_LENGTH, MODEL_SYSTEM_PROMPT
from run_helper import banner, follow

logger = logging.getLogger(__name__)

# Explicit multi-turn Gemma chat template + sampling params. Stock gemma4 uses
# `RENDERER gemma4` / `PARSER gemma4` directives that Ollama ties to the stock
# blob digest and rejects for derived models, so we write our own frame. Without
# this, Unsloth's GGUF export produces a file whose chat template metadata isn't
# reliably propagated and Ollama's default frame doesn't match how the model was
# trained — the model generates but never emits <end_of_turn>.
MODELFILE_TEMPLATE = '''FROM __GGUF_PATH__
TEMPLATE """{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
<start_of_turn>{{ if eq .Role "user" }}user
{{ else }}model
{{ end }}
{{- if and (eq .Role "user") (eq $i 0) $.System }}{{ $.System }}

{{ end }}{{ .Content }}<end_of_turn>
{{ if $last }}<start_of_turn>model
{{ end }}
{{- end }}"""
PARAMETER stop "<end_of_turn>"
PARAMETER stop "<start_of_turn>"
PARAMETER temperature 1.0
PARAMETER top_k 64
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.0
PARAMETER num_ctx 32768
'''


def _build_modelfile(gguf_path, system_prompt):
    body = MODELFILE_TEMPLATE.replace("__GGUF_PATH__", str(gguf_path))
    return body.rstrip() + f'\n\nSYSTEM """{system_prompt}"""\n'


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

    # Exclude BF16 intermediates and vision projector (mmproj). For sharded
    # models pick the primary shard (00001-of-N); fall back to any non-shard
    # file; last resort: whatever is first.
    candidates = [f for f in gguf_files if "BF16" not in f.name and "mmproj" not in f.name]
    gguf_path = (
        next((f for f in candidates if "00001-of" in f.name), None)
        or next((f for f in candidates if "-of-" not in f.name), None)
        or (candidates[0] if candidates else gguf_files[0])
    ).resolve()

    modelfile_path = output_path / "Modelfile"
    modelfile_path.write_text(_build_modelfile(gguf_path, MODEL_SYSTEM_PROMPT))

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
        log_path = Path("logs") / "export.log"
        log_path.parent.mkdir(exist_ok=True)
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
        follow(proc, log_path, "export")
    else:
        banner("EXPORT — STARTING")
        export(adapter_path, output_path, args.quant, args.model_name)
        banner("EXPORT — DONE")
