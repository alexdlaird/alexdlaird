.PHONY: all install nopyc clean pretrain-tone generate-data train export register smoke-test

LOG_DIR := logs

SHELL := /usr/bin/env bash
PYTHON_BIN    ?= python
PROJECT_VENV  ?= venv
TOOLS_CORTEX  ?= $(HOME)/Developer/alexdlaird/tools/cortex
QUANT         ?= q4_k_m
MODEL_NAME    ?= cortex
INCLUDE_TESTS ?= false

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

pretrain-tone: install
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(TOOLS_CORTEX)/finetune/fetch_blog.py; \
		python $(TOOLS_CORTEX)/finetune/pretrain_tone.py; \
	)

generate-data: install
	@mkdir -p $(LOG_DIR)
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(TOOLS_CORTEX)/finetune/generate_data.py $(if $(filter true,$(INCLUDE_TESTS)),--include-tests,); \
	)

train: install
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(TOOLS_CORTEX)/finetune/train.py; \
	)

export: install
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(TOOLS_CORTEX)/finetune/export.py --quant $(QUANT) --model-name $(MODEL_NAME); \
	)

register:
	@# Clear Ollama's cached manifest/blob digest for this model; otherwise
	@# re-registering an overwritten GGUF trips "digest mismatch".
	-@ollama rm $(MODEL_NAME) 2>/dev/null
	ollama create $(MODEL_NAME) -f $$($(PROJECT_VENV)/bin/python -c \
		"from config import FINETUNE_OUTPUT_DIR; print(FINETUNE_OUTPUT_DIR / 'gguf' / 'Modelfile')")
	@$(MAKE) --no-print-directory smoke-test
	docker restart pipelines

# Fails if the model doesn't terminate cleanly on a trivial prompt — the
# canonical signature of a broken chat template or missing stop token.
smoke-test:
	@echo "--> smoke-testing $(MODEL_NAME) ..."
	@result=$$(curl -sS --max-time 60 http://localhost:11434/api/chat -d '{ \
		"model":"$(MODEL_NAME)", \
		"messages":[{"role":"user","content":"Reply with the single word: hello"}], \
		"stream":false}'); \
	done_reason=$$(printf '%s' "$$result" | $(PROJECT_VENV)/bin/python -c \
		'import json,sys; print(json.load(sys.stdin).get("done_reason",""))'); \
	if [ "$$done_reason" != "stop" ]; then \
		echo "!!> smoke test FAILED: done_reason='$$done_reason' (expected 'stop')"; \
		printf '%s' "$$result" | head -c 500; echo; \
		exit 1; \
	fi; \
	echo "--> smoke test passed (done_reason=stop)"
