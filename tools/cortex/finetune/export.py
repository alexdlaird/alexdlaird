__copyright__ = "Copyright (c) 2026 Alex Laird"
__license__ = "MIT"

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from config import FINETUNE_OUTPUT_DIR, MODEL_SYSTEM_PROMPT
from run_helper import banner, follow

logger = logging.getLogger(__name__)

# llama.cpp checkout that ships its own convert_hf_to_gguf.py + gguf-py + llama-quantize
# binary, all from the same commit so they're internally consistent. Calling these
# directly is the structural fix for unsloth_zoo's GGUF path, which downloads
# convert_hf_to_gguf.py from llama.cpp master HEAD at runtime and breaks whenever
# upstream adds an arch enum the locally-installed gguf doesn't yet expose.
LLAMA_CPP_DIR = Path("~/.unsloth/llama.cpp").expanduser()

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
PARAMETER temperature 0.7
PARAMETER top_k 64
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.0
PARAMETER num_ctx 32768
PARAMETER num_predict 8192
'''

# Agent overlay Modelfile — FROM the chat model, with tool-calling-friendly sampling
# params and the agent system prompt baked in. Overrides the chat params inherited
# from the base.
AGENT_OVERLAY_TEMPLATE = '''FROM __BASE_MODEL__
PARAMETER stop "<end_of_turn>"
PARAMETER stop "<start_of_turn>"
PARAMETER temperature 0.2
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.05
PARAMETER num_predict 8192
'''


def _build_modelfile(gguf_path, system_prompt):
    body = MODELFILE_TEMPLATE.replace("__GGUF_PATH__", str(gguf_path))
    return body.rstrip() + f'\n\nSYSTEM """{system_prompt}"""\n'


def _build_agent_overlay(base_model, agent_prompt):
    body = AGENT_OVERLAY_TEMPLATE.replace("__BASE_MODEL__", base_model)
    return body.rstrip() + f'\n\nSYSTEM """{agent_prompt}"""\n'


def _convert_hf_to_gguf(model_dir, outfile):
    cmd = [
        sys.executable, str(LLAMA_CPP_DIR / "convert_hf_to_gguf.py"),
        str(model_dir), "--outfile", str(outfile), "--outtype", "bf16",
    ]
    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _quantize_gguf(infile, outfile, quant):
    cmd = [str(LLAMA_CPP_DIR / "llama-quantize"), str(infile), str(outfile), quant]
    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def export(adapter_path, output_path, quant, model_name):
    output_path.mkdir(parents=True, exist_ok=True)

    # Clear GGUFs from prior runs so the Modelfile's FROM always points at a
    # freshly-hashed file. Without this, Ollama sees a digest mismatch.
    for stale in output_path.glob("*.gguf"):
        logger.info(f"Removing stale GGUF: {stale}")
        stale.unlink()

    bf16_path = output_path / f"{model_name}.bf16.gguf"
    final_path = output_path / f"{model_name}.{quant.upper()}.gguf"

    logger.info(f"Converting {adapter_path} -> {bf16_path}")
    _convert_hf_to_gguf(adapter_path, bf16_path)

    logger.info(f"Quantizing {bf16_path} -> {final_path} ({quant})")
    _quantize_gguf(bf16_path, final_path, quant)

    bf16_path.unlink()

    gguf_path = final_path.resolve()

    modelfile_path = output_path / "Modelfile"
    modelfile_path.write_text(_build_modelfile(gguf_path, MODEL_SYSTEM_PROMPT))

    agent_prompt_path = Path(__file__).parent.parent / "prompts" / "agent_system_prompt.txt"
    agent_prompt = agent_prompt_path.read_text().strip()
    agent_modelfile_path = output_path / "Modelfile.agent"
    agent_modelfile_path.write_text(_build_agent_overlay(model_name, agent_prompt))

    logger.info(f"Modelfile written to {modelfile_path}")
    logger.info(f"Agent overlay Modelfile written to {agent_modelfile_path}")
    logger.info("")
    logger.info("To register with Ollama, run:")
    logger.info(f"  ollama create {model_name} -f {modelfile_path}")
    logger.info(f"  ollama create {model_name}-agent -f {agent_modelfile_path}")


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
