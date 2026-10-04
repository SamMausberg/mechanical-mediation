PYTHON ?= python3
.PHONY: all paper calculate test intervals reproduce clean
all: paper
paper:
	./build.sh
calculate:
	@mkdir -p results
	$(PYTHON) code/calculate.py > results/calculation.log
test:
	@mkdir -p results
	$(PYTHON) -m unittest discover -s code -p 'test_*.py' -v > results/tests.log 2>&1
	@tail -5 results/tests.log
intervals:
	$(PYTHON) code/design_interval.py > results/design_interval.log
	$(PYTHON) code/interval_check.py > results/null_interval.log
	@tail -1 results/design_interval.log
	@tail -1 results/null_interval.log
reproduce: calculate test intervals paper
clean:
	rm -f paper/main.aux paper/main.log paper/main.out paper/mainNotes.bib
