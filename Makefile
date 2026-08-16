PYTHON := .venv/bin/python

.PHONY: setup verify play

setup:
	scripts/setup.sh Release

verify:
	$(PYTHON) -m mars_rover_env.tools.doctor

play:
	$(PYTHON) -m mars_rover_env.tools.play
