PYTHON ?= python3
PIP ?= pip3

.PHONY: setup install dev run-logreg run-tree clean reports

setup:
	$(PIP) install -e .

install:
	$(PIP) install -e .[dev]

run-logreg:
	$(PYTHON) -m src.train --model logreg --generate --n_rows 20000 --seed 42

run-tree:
	$(PYTHON) -m src.train --model tree --generate --n_rows 20000 --seed 42

reports:
	@echo "Reports in reports/ and figures in reports/figures"

clean:
	rm -rf .pytest_cache .mypy_cache build dist *.egg-info
	rm -f reports/metrics.json reports/fraud_results.csv
	rm -f data/transactions.csv


