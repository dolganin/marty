PYTHON ?= python3
MODEL ?=
ARGS ?=

.PHONY: build score gui play

build:
	$(PYTHON) -m pip install --editable . --no-build-isolation
	$(PYTHON) -m pip install --editable evaluator --no-build-isolation

score:
	mars-rover-score $(MODEL) $(ARGS)

gui:
	$(PYTHON) -m pip install --editable gui --no-build-isolation

play: gui
	mars-rover-organizer-play $(ARGS)
