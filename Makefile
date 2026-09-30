PYTHON ?= python

.PHONY: reproduce simulate plot test

reproduce: simulate plot

simulate:
	$(PYTHON) -m src.run_experiments

plot:
	$(PYTHON) -m src.plot_results

test:
	$(PYTHON) -m pytest
