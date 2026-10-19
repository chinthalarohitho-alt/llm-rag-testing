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

# Step 1 compares v1 vs v2. v1 is the "before" prompt and is EXPECTED to fail
# some checks, so the leading "-" tells make to keep going.
# Step 2 is the real gate: the v2 prompt must pass every check (re-uses cached answers).
promptfoo:       ## Promptfoo: compare prompt v1 vs v2, then require v2 to pass
	-$(PROMPTFOO) eval -c promptfoo/promptfooconfig.yaml
	$(PROMPTFOO) eval -c promptfoo/promptfooconfig.yaml --filter-prompts support_v2

promptfoo-rag:   ## Promptfoo: end-to-end test of the RAG app
	PROMPTFOO_PYTHON=$(PY) $(PROMPTFOO) eval -c promptfoo/rag_eval.yaml

report:          ## Open the Promptfoo results dashboard in your browser
	$(PROMPTFOO) view

test-all: test-offline test-deepeval promptfoo promptfoo-rag  ## Run everything
