.PHONY: all install nopyc clean fetch-blog pretrain-tone pretrain-tone-resume generate-data generate-data-sample train train-merge export register

SHELL := /usr/bin/env bash
PYTHON_BIN    ?= python
PROJECT_VENV  ?= venv
TOOLS_CORTEX  ?= $(HOME)/Developer/alexdlaird/tools/cortex
QUANT         ?= q4_k_m
MODEL_NAME    ?= cortex-ft

all: install

venv:
	$(PYTHON_BIN) -m pip install virtualenv --user
	$(PYTHON_BIN) -m virtualenv $(PROJECT_VENV)

install: venv
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		CUDA_VER=$$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | tr -d '.'); \
		pip install torch --index-url "https://download.pytorch.org/whl/cu$${CUDA_VER}"; \
		pip install -r $(TOOLS_CORTEX)/finetune/requirements.txt; \
	)

nopyc:
	find . -name '*.pyc' | xargs rm -f || true
	find . -name __pycache__ | xargs rm -rf || true

clean: nopyc
	rm -rf $(PROJECT_VENV)

fetch-blog: install
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(TOOLS_CORTEX)/finetune/fetch_blog.py; \
	)

pretrain-tone: install
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(TOOLS_CORTEX)/finetune/pretrain_tone.py; \
	)

pretrain-tone-resume: install
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(TOOLS_CORTEX)/finetune/pretrain_tone.py --resume; \
	)

generate-data: install
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(TOOLS_CORTEX)/finetune/generate_data.py; \
	)

generate-data-sample: install
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(TOOLS_CORTEX)/finetune/generate_data.py --sample-every 10 --limit 200; \
	)

train: install
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(TOOLS_CORTEX)/finetune/train.py; \
	)

train-merge: install
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(TOOLS_CORTEX)/finetune/train.py --merge; \
	)

export: install
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(TOOLS_CORTEX)/finetune/export.py --quant $(QUANT) --model-name $(MODEL_NAME); \
	)

register:
	ollama create $(MODEL_NAME) -f $$($(PROJECT_VENV)/bin/python -c \
		"from config import FINETUNE_OUTPUT_DIR; print(FINETUNE_OUTPUT_DIR / 'gguf' / 'Modelfile')")
