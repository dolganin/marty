PYTHON ?= python3
ARGS ?=

.PHONY: build train check gui play gui-test

build:
	$(PYTHON) -m pip install --editable . --no-build-isolation
	$(PYTHON) -m pip install --editable agents --no-build-isolation

train:
	mars-rover-train $(ARGS)

check:
	docker build --tag mars-rover-submission-check .

gui:
	$(PYTHON) -m pip install --editable gui --no-build-isolation

play: gui
	mars-rover-play $(ARGS)

gui-test:
	$(PYTHON) -m unittest discover -s gui/tests -v
