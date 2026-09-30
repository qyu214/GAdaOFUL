PYTHON ?= python

.PHONY: reproduce simulate plot

reproduce: simulate plot

simulate:
	$(PYTHON) -m src.run_experiments

plot:
	$(PYTHON) -m src.plot_results
