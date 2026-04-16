.PHONY: all install nopyc clean fetch-blog pretrain-tone pretrain-tone-resume pretrain-tone-bg generate-data generate-data-sample generate-data-bg train train-bg train-merge train-merge-bg export export-bg register

LOG_DIR := logs

SHELL := /usr/bin/env bash
PYTHON_BIN    ?= python
PROJECT_VENV  ?= venv
TOOLS_CORTEX  ?= $(HOME)/Developer/alexdlaird/tools/cortex
QUANT         ?= q4_k_m
MODEL_NAME    ?= cortex

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

pretrain-tone-bg: install
	@mkdir -p $(LOG_DIR)
	@nohup $(MAKE) pretrain-tone > $(LOG_DIR)/pretrain-tone.log 2>&1 &
	@echo "--> pretrain-tone running in background (PID $$!), tailing log ..."
	@tail -f $(LOG_DIR)/pretrain-tone.log

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

generate-data-bg: install
	@mkdir -p $(LOG_DIR)
	@nohup $(MAKE) generate-data > $(LOG_DIR)/generate-data.log 2>&1 &
	@echo "--> generate-data running in background (PID $$!), tailing log ..."
	@tail -f $(LOG_DIR)/generate-data.log

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

train-bg: install
	@mkdir -p $(LOG_DIR)
	@nohup $(MAKE) train > $(LOG_DIR)/train.log 2>&1 &
	@echo "--> train running in background (PID $$!), tailing log ..."
	@tail -f $(LOG_DIR)/train.log

train: install
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(TOOLS_CORTEX)/finetune/train.py; \
	)

train-merge-bg: install
	@mkdir -p $(LOG_DIR)
	@nohup $(MAKE) train-merge > $(LOG_DIR)/train-merge.log 2>&1 &
	@echo "--> train-merge running in background (PID $$!), tailing log ..."
	@tail -f $(LOG_DIR)/train-merge.log

train-merge: install
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(TOOLS_CORTEX)/finetune/train.py --merge; \
	)

export-bg: install
	@mkdir -p $(LOG_DIR)
	@nohup $(MAKE) export > $(LOG_DIR)/export.log 2>&1 &
	@echo "--> export running in background (PID $$!), tailing log ..."
	@tail -f $(LOG_DIR)/export.log

export: install
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(TOOLS_CORTEX)/finetune/export.py --quant $(QUANT) --model-name $(MODEL_NAME); \
	)

register:
	ollama create $(MODEL_NAME) -f $$($(PROJECT_VENV)/bin/python -c \
		"from config import FINETUNE_OUTPUT_DIR; print(FINETUNE_OUTPUT_DIR / 'gguf' / 'Modelfile')")
