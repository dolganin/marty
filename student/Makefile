PYTHON ?= python3
ARGS ?=

.PHONY: build train

build:
	$(PYTHON) -m pip install --editable . --no-build-isolation
	$(PYTHON) -m pip install --editable agents --no-build-isolation

train:
	mars-rover-train $(ARGS)
