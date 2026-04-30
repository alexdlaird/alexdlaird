.PHONY: all pull pull-blog install nopyc clean ingest pretrain-tone generate-data train export register unregister smoke-test probe setup-base-agent pipeline pipeline-bg

LOG_DIR := logs

SHELL := /usr/bin/env bash
PYTHON_BIN      ?= python
PROJECT_VENV    ?= venv
ALEXDLAIRD_DIR  ?= $(HOME)/Developer/alexdlaird
PRIVATE_DEV_DIR ?= $(HOME)/Developer/private-dev
TOOLS_CORTEX    ?= $(ALEXDLAIRD_DIR)/tools/cortex
QUANT           ?= q4_k_m
MODEL_NAME      ?= cortex
INCLUDE_TESTS   ?= false
SAMPLE_EVERY    ?= 1
LIMIT           ?= 0
PIPELINE_SESSION ?= cortex-retrain

all: install

pull:
	@git -C $(ALEXDLAIRD_DIR) pull --ff-only
	@git -C $(PRIVATE_DEV_DIR) pull --ff-only

pull-blog:
	@git -C $(HOME)/Developer/alexdlaird.github.io pull --ff-only

venv:
	$(PYTHON_BIN) -m pip install virtualenv --user
	$(PYTHON_BIN) -m virtualenv $(PROJECT_VENV)

install: pull venv
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		CUDA_VER=$$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | tr -d '.'); \
		pip install torch --index-url "https://download.pytorch.org/whl/cu$${CUDA_VER}"; \
		pip install -r $(TOOLS_CORTEX)/finetune/requirements.txt; \
		if [ -d $(HOME)/.unsloth/llama.cpp/gguf-py ]; then \
			pip install --force-reinstall --no-deps $(HOME)/.unsloth/llama.cpp/gguf-py; \
		fi; \
	)

nopyc:
	find . -name '*.pyc' | xargs rm -f || true
	find . -name __pycache__ | xargs rm -rf || true

clean: nopyc
	rm -rf $(PROJECT_VENV)

ingest:
	@$(MAKE) --no-print-directory -C ../rag ingest

pretrain-tone: install pull-blog
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(TOOLS_CORTEX)/finetune/pretrain_tone.py; \
	)

generate-data: install
	@mkdir -p $(LOG_DIR)
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(TOOLS_CORTEX)/finetune/generate_data.py \
			--sample-every $(SAMPLE_EVERY) \
			$(if $(filter-out 0,$(LIMIT)),--limit $(LIMIT),) \
			$(if $(filter true,$(INCLUDE_TESTS)),--include-tests,); \
	)

train: install
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(TOOLS_CORTEX)/finetune/train.py; \
	)

train-merge: install
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(TOOLS_CORTEX)/finetune/train.py --merge-only; \
	)

export: install
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(TOOLS_CORTEX)/finetune/export.py --quant $(QUANT) --model-name $(MODEL_NAME); \
	)

setup-base-agent: pull
	@# One-off bootstrap: create gemma4-agent overlay + cortex-agent alias.
	@# Idempotent. Safe to re-run on an already-configured machine.
	$(TOOLS_CORTEX)/bin/setup-base-agent
	@$(MAKE) --no-print-directory probe

unregister:
	-@ollama rm $(MODEL_NAME) 2>/dev/null
	-@ollama rm $(MODEL_NAME)-agent 2>/dev/null
	@# Re-alias the agent name back to the base overlay so pi/opencode keep
	@# working with the same client config after the fine-tune is removed. The
	@# chat-facing $(MODEL_NAME) stays absent; pipeline.py's _resolve_model()
	@# falls back to plain gemma4 on the chat path.
	ollama cp gemma4-agent $(MODEL_NAME)-agent
	docker restart pipelines

register:
	@# Clear Ollama's cached manifest/blob digest for this model; otherwise
	@# re-registering an overwritten GGUF trips "digest mismatch".
	-@ollama rm $(MODEL_NAME) 2>/dev/null
	-@ollama rm $(MODEL_NAME)-agent 2>/dev/null
	ollama create $(MODEL_NAME) -f $$($(PROJECT_VENV)/bin/python -c \
		"from config import FINETUNE_OUTPUT_DIR; print(FINETUNE_OUTPUT_DIR / 'gguf' / 'Modelfile')")
	ollama create $(MODEL_NAME)-agent -f $$($(PROJECT_VENV)/bin/python -c \
		"from config import FINETUNE_OUTPUT_DIR; print(FINETUNE_OUTPUT_DIR / 'gguf' / 'Modelfile.agent')")
	@$(MAKE) --no-print-directory smoke-test
	@$(MAKE) --no-print-directory probe
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

# Tool-calling reliability probe against $(MODEL_NAME)-agent. Verifies the agent
# overlay (gemma4-agent or real cortex-agent) emits structured tool_calls — the
# failure mode we saw with temp=1.0 baked into the base Modelfile.
probe: pull
	@echo "--> probing $(MODEL_NAME)-agent for tool-calling reliability ..."
	$(TOOLS_CORTEX)/tests/tool_call_probe.sh $(MODEL_NAME)-agent 10 http://localhost:11434

# Full unattended retrain: ingest → pretrain-tone → generate-data → train → export → register.
# Spawns a detached tmux session running pipeline-bg and tails the log. Ctrl-C
# stops the tail; the tmux session keeps running. Reattach via:
#   tmux attach -t $(PIPELINE_SESSION)
# Override SAMPLE_EVERY for a faster validation pass: make pipeline SAMPLE_EVERY=10
pipeline:
	@mkdir -p $(LOG_DIR)
	@if tmux has-session -t $(PIPELINE_SESSION) 2>/dev/null; then \
		echo "--> tmux session '$(PIPELINE_SESSION)' already exists; tailing existing log."; \
	else \
		: > $(LOG_DIR)/pipeline.log; \
		tmux new-session -d -s $(PIPELINE_SESSION) "$(MAKE) --no-print-directory pipeline-bg > $(LOG_DIR)/pipeline.log 2>&1"; \
		echo "--> Started pipeline in detached tmux session '$(PIPELINE_SESSION)'."; \
	fi
	@echo "--> Tailing $(LOG_DIR)/pipeline.log (Ctrl-C to stop tailing; pipeline keeps running)."
	@echo "--> Reattach later with: tmux attach -t $(PIPELINE_SESSION)"
	@PANE_PID=$$(tmux list-panes -t $(PIPELINE_SESSION) -F "#{pane_pid}" 2>/dev/null | head -1); \
	  exec tail -f $${PANE_PID:+--pid=$$PANE_PID} $(LOG_DIR)/pipeline.log

# The actual chain. Halts on first failure. Ollama models are evicted between
# Ollama-using and HuggingFace-using stages to avoid VRAM contention with the
# 31B Gemma 4 load. Invoked inside the tmux session by `make pipeline`.
pipeline-bg:
	$(MAKE) --no-print-directory ingest
	-@ollama stop nomic-embed-text 2>/dev/null
	$(MAKE) --no-print-directory pretrain-tone
	$(MAKE) --no-print-directory generate-data
	-@ollama stop gemma4 2>/dev/null
	$(MAKE) --no-print-directory train
	$(MAKE) --no-print-directory export
	$(MAKE) --no-print-directory register
