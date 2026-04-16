.PHONY: all install nopyc clean ingest ingest-bg ingest-docs ingest-docs-bg rebuild rebuild-bg search

LOG_DIR := logs

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

ingest-bg: install
	@mkdir -p $(LOG_DIR)
	@nohup $(MAKE) ingest > $(LOG_DIR)/ingest.log 2>&1 &
	@echo "--> ingest running in background (PID $$!), tailing log ..."
	@tail -f $(LOG_DIR)/ingest.log

ingest: install
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(TOOLS_CORTEX)/ingest/repos.py; \
	)

rebuild-bg: install
	@mkdir -p $(LOG_DIR)
	@nohup $(MAKE) rebuild > $(LOG_DIR)/rebuild.log 2>&1 &
	@echo "--> rebuild running in background (PID $$!), tailing log ..."
	@tail -f $(LOG_DIR)/rebuild.log

rebuild: install
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(TOOLS_CORTEX)/ingest/repos.py --rebuild; \
	)

ingest-docs-bg: install
	@mkdir -p $(LOG_DIR)
	@nohup $(MAKE) ingest-docs > $(LOG_DIR)/ingest-docs.log 2>&1 &
	@echo "--> ingest-docs running in background (PID $$!), tailing log ..."
	@tail -f $(LOG_DIR)/ingest-docs.log

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
