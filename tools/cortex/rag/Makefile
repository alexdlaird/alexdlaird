.PHONY: all install nopyc clean ingest search refresh restart

SHELL := /usr/bin/env bash
PYTHON_BIN ?= python
PROJECT_VENV ?= venv
TOOLS_CORTEX ?= $(HOME)/Developer/alexdlaird/tools/cortex
TOP_K ?= 5
INCLUDE_TESTS ?= false

all: install

venv:
	$(PYTHON_BIN) -m pip install virtualenv --user
	$(PYTHON_BIN) -m virtualenv $(PROJECT_VENV)

install: venv
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		pip install -r $(TOOLS_CORTEX)/rag/requirements.txt; \
	)

nopyc:
	find . -name '*.pyc' | xargs rm -f || true
	find . -name __pycache__ | xargs rm -rf || true

clean: nopyc
	rm -rf $(PROJECT_VENV)

ingest: install
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(TOOLS_CORTEX)/ingest/repos.py --rebuild; \
		python $(TOOLS_CORTEX)/ingest/docs.py; \
	)

search: install
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(TOOLS_CORTEX)/query/search.py --top-k $(TOP_K) $(if $(filter true,$(INCLUDE_TESTS)),--include-tests,) "$(QUERY)"; \
	)

refresh:
	docker cp $(TOOLS_CORTEX)/pipeline/pipeline.py pipelines:/app/pipelines/pipeline.py
	docker restart pipelines

restart:
	docker restart $$(docker ps -aq --filter "network=cortex")
