VENV ?= .venv
PY := $(VENV)/bin/python

.PHONY: install fmt lint test run scan

install:
	python3 -m venv $(VENV)
	$(PY) -m pip install -r requirements-dev.txt

fmt:
	$(PY) -m ruff format pricetracker tests
	$(PY) -m ruff check --fix pricetracker tests

lint:
	$(PY) -m ruff format --check pricetracker tests
	$(PY) -m ruff check pricetracker tests

test:
	$(PY) -m pytest -q

run:
	$(PY) -m pricetracker

scan:
	$(PY) -m pricetracker scan
