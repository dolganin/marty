PYTHON := .venv/bin/python

.PHONY: setup test play

setup:
	scripts/setup.sh Release

test:
	$(PYTHON) -m pytest environment/tests baselines/tests -q

play:
	$(PYTHON) -m mars_rover_env.tools.play
