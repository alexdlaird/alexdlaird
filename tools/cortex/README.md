# cortex 🤖

`cortex` is my AI playground, housing a RTX 5090 to melt the shelf it's on in my office closet.
It uses RAG (Retrieval-Augmented Generation) tooling to power a local coding assistant, built on
[Ollama](https://ollama.com) and [Qdrant](https://qdrant.tech).

## Prerequisites

Provision the full AI stack first using [`dev-init-ai`](../init/dev-init-ai) from this repo. This
installs the dependencies pulls the embedding model (`nomic-embed-text`), and starts the Docker
containers.

## Setup

Create a private directory with a `config.py` and a `Makefile` — these are never checked in to
this repo as they contain personal repo paths and settings.

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

**`Makefile`:**

```makefile
.PHONY: all install nopyc clean ingest ingest-docs rebuild search

SHELL := /usr/bin/env bash
PYTHON_BIN ?= python
PROJECT_VENV ?= venv
TOOLS_CORTEX ?= $(HOME)/Developer/alexdlaird/tools/cortex
TOP_K ?= 5

all: install

venv:
	$(PYTHON_BIN) -m pip install virtualenv --user
	$(PYTHON_BIN) -m virtualenv $(PROJECT_VENV)

install: venv
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		pip install -r $(TOOLS_CORTEX)/requirements.txt; \
	)

nopyc:
	find . -name '*.pyc' | xargs rm -f || true
	find . -name __pycache__ | xargs rm -rf || true

clean: nopyc
	rm -rf $(PROJECT_VENV)

ingest: install
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(TOOLS_CORTEX)/ingest/repos.py; \
	)

rebuild: install
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(TOOLS_CORTEX)/ingest/repos.py --rebuild; \
	)

ingest-docs: install
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(TOOLS_CORTEX)/ingest/docs.py; \
	)

search: install
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(TOOLS_CORTEX)/query/search.py --top-k $(TOP_K) "$(QUERY)"; \
	)
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
