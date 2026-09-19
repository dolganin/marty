PYTHON ?= python3
ARGS ?=

.PHONY: build play

build:
	$(PYTHON) -m pip install --editable . --no-build-isolation

play:
	mars-rover-play $(ARGS)
