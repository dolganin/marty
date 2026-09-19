PYTHON ?= python3
MODEL ?=
ARGS ?=

.PHONY: build score

build:
	$(PYTHON) -m pip install --editable . --no-build-isolation
	$(PYTHON) -m pip install --editable evaluator --no-build-isolation

score:
	mars-rover-score $(MODEL) $(ARGS)
