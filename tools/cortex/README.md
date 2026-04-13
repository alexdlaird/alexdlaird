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
COLLECTION_NAME = "my-rag"

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
# Ingest repos into the vector index
make ingest

# Rebuild the index from scratch (drops and recreates the collection)
make rebuild

# Ingest standalone docs/specs (paths defined in DOC_PATHS in config.py)
make ingest-docs

# Search the index
make search QUERY="how does the auth middleware work"
make search QUERY="database migration pattern" TOP_K=10
```

Once indexed, Open WebUI at [http://localhost:3000](http://localhost:3000) connects to Ollama
for interactive use as a coding assistant.

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

QDRANT_URL        = "http://localhost:6333"
OLLAMA_BASE_URL   = "http://localhost:11434"
COLLECTION_NAME   = "my-rag"
OLLAMA_CHAT_MODEL = "gemma4"

# Requires accepting the license at huggingface.co/google/gemma-4-27b-it
HF_MODEL_ID = "google/gemma-4-27b-it"

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
# Validate Q&A generation quality on a small sample before committing to the full run
make generate-data-sample

# Generate training data from the full RAG corpus
make generate-data

# Train the QLoRA adapter (iterate here; use --resume to continue a run)
make train

# Train and merge adapter into full weights (required before export)
make train-merge

# Export to GGUF and generate Modelfile
make export

# Register the fine-tuned model with Ollama
make register
```
