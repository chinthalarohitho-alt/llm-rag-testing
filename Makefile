# One-word commands for everything. Run `make help` to see them.
PY = .venv/bin/python
PYTHON ?= $(shell command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)
PROMPTFOO = PROMPTFOO_DISABLE_TELEMETRY=1 npx -y promptfoo@latest
MODEL = gemma3:4b
export DEEPEVAL_TELEMETRY_OPT_OUT = 1

.PHONY: help setup models test-offline test-deepeval promptfoo promptfoo-rag report test-all

help:            ## Show all commands
	@grep -E '^[a-z-]+:.*##' Makefile | awk -F':.*## ' '{printf "  make %-15s %s\n", $$1, $$2}'

setup: models    ## Create virtual env, install dependencies, download the model
	$(PYTHON) -m venv .venv
	.venv/bin/pip install -r requirements.txt

models:          ## Download the local Ollama model
	ollama pull $(MODEL)

test-offline:    ## Fast tests, no model needed
	$(PY) -m pytest tests/offline_tests

test-deepeval:   ## DeepEval: quality, RAG and safety tests
	.venv/bin/deepeval test run tests/deepeval_tests

# v1 is the "before" prompt and is EXPECTED to fail 2 of 10 checks, so we pass at 80%.
promptfoo:       ## Promptfoo: compare prompt v1 vs v2
	PROMPTFOO_PASS_RATE_THRESHOLD=80 $(PROMPTFOO) eval -c promptfoo/promptfooconfig.yaml

promptfoo-rag:   ## Promptfoo: end-to-end test of the RAG app
	PROMPTFOO_PYTHON=$(PY) $(PROMPTFOO) eval -c promptfoo/rag_eval.yaml

report:          ## Open the Promptfoo results dashboard in your browser
	$(PROMPTFOO) view

test-all: test-offline test-deepeval promptfoo promptfoo-rag  ## Run everything
