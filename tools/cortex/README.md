# cortex 🤖

`cortex` is my AI playground, housing a RTX 5090 to melt the shelf it's on in my office closet.
It uses RAG (Retrieval-Augmented Generation) tooling and fine-tuning to power a local coding assistant,
built on [Ollama](https://ollama.com) and [Qdrant](https://qdrant.tech).

## Prerequisites

Provision the full AI stack first using [`dev-init-ai`](../init/dev-init-ai) from this repo. This
installs the dependencies pulls the embedding model (`nomic-embed-text`), and starts the Docker
containers.

## Setup

Create a private directory with a `config.py` — never check this in to any public repo. The
`Makefile` is provisioned into place automatically by [`dev-init-ai`](../init/dev-init-ai).

**`config.py`:**

```python
from pathlib import Path

DEVELOPER_DIR = Path.home() / "Developer"

QDRANT_URL = "http://localhost:6333"
OLLAMA_BASE_URL = "http://localhost:11434"

EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "cortex"

CODE_EXTENSIONS = [
    # Application code
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".scala", ".dart", ".sh",
    # Web/styling
    ".html", ".css", ".scss",
    # Config/build
    ".yaml", ".yml", ".toml", ".xml", ".properties", ".gradle", ".conf",
    # Infrastructure
    ".tf", ".hcl",
]
DOC_EXTENSIONS = [".md", ".rst", ".txt"]

# Local repo paths (relative to DEVELOPER_DIR) to ingest
REPOS = [
    "my-project",
    "another-project",
]

# Paths to standalone doc/spec files or directories
DOC_PATHS: list[str] = []
```

## Usage

Run all commands from your config directory:

```bash
# Ingest repos and docs into the vector index (always rebuilds from scratch)
make ingest

# Search the index
make search QUERY="how does the auth middleware work"
make search QUERY="database migration pattern" TOP_K=10
```

## Models

Each base (`gemma4` or the fine-tuned `cortex`) is registered twice in Ollama — once
for chat, once for agent use:

| Model          | Path  | Purpose                                          | Used by              |
|----------------|-------|--------------------------------------------------|----------------------|
| `gemma4`       | chat  | Google base, untouched                           | pipeline fallback    |
| `cortex`       | chat  | Fine-tuned, baked with chat-oriented sampling    | pipeline preferred   |
| `gemma4-agent` | agent | `FROM gemma4` + tool-calling params + agent prompt | pi/opencode fallback |
| `cortex-agent` | agent | `FROM cortex` overlay (or alias to `gemma4-agent` pre-fine-tune) | pi/opencode preferred |

- **Open WebUI** at [http://localhost:3000](http://localhost:3000) goes through the RAG
  pipeline, which picks the best chat-path model available (`cortex` if registered,
  else `gemma4`).
- **CLI coding agents** (pi, opencode) point at `cortex-agent` directly — no RAG, since
  they have filesystem and tool access of their own. When there's no fine-tune yet,
  `cortex-agent` is an alias to `gemma4-agent`; `make register` after a fine-tune
  replaces the alias with the real overlay. Clients never need to change targets.

## Fine-tuning

Fine-tuning uses the RAG corpus to generate synthetic training data, then trains a QLoRA adapter
on top of the base model via [Unsloth](https://github.com/unslothai/unsloth), and exports the
result as a GGUF for Ollama.

Create a separate private directory for fine-tuning with its own `config.py`. The `Makefile` is
provisioned into place automatically by [`dev-init-ai`](../init/dev-init-ai).

**`config.py`:**

```python
from pathlib import Path

FINETUNE_DATA_DIR   = Path.home() / "cortex-finetune" / "data"
FINETUNE_OUTPUT_DIR = Path.home() / "cortex-finetune" / "output"
BLOG_OUTPUT_DIR     = FINETUNE_DATA_DIR / "blog"

BLOG_SITEMAP_URL = "https://www.example.com/sitemap.xml"
SEED_DATA_PATHS  = []  # list of Path objects to seed .jsonl files

TRAINING_SYSTEM_PROMPT = "You are a senior software engineering assistant ..."
MODEL_SYSTEM_PROMPT = "You are a direct, technical coding assistant ..."

QDRANT_URL        = "http://localhost:6333"
OLLAMA_BASE_URL   = "http://localhost:11434"
COLLECTION_NAME   = "cortex"
OLLAMA_CHAT_MODEL = "gemma4"

# Requires accepting the license at huggingface.co/google/gemma-4-31B-it
HF_MODEL_ID = "google/gemma-4-31B-it"

MAX_SEQ_LENGTH = 2048

LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

TRAIN_BATCH_SIZE            = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE               = 2e-4
NUM_EPOCHS                  = 3
WARMUP_RATIO                = 0.05
```

### Usage

```bash
# One-off bootstrap: register the gemma4-agent overlay on Ollama and alias
# cortex-agent to it so pi/opencode work against the base model. Idempotent —
# dev-init-ai runs this automatically, but it's safe to re-run on its own.
make setup-base-agent

# Fetch blog posts and run tone pre-training
make pretrain-tone

# Generate training data from the RAG corpus (interactive; runs in background)
make generate-data

# Train the QLoRA adapter and merge into full weights
make train

# Export to GGUF and generate Modelfile (writes both Modelfile and Modelfile.agent)
make export

# Register the fine-tuned model with Ollama (creates both cortex and cortex-agent,
# runs smoke-test against cortex and tool-calling probe against cortex-agent, then
# restarts the pipelines container to pick up the new model)
make register

# Tool-calling reliability probe against cortex-agent (also run automatically as
# part of `make register` and `make setup-base-agent`)
make probe
```

`make unregister` removes the fine-tuned `cortex` and `cortex-agent`, then re-aliases
`cortex-agent` back to `gemma4-agent` so pi/opencode keep working against the base
overlay.
