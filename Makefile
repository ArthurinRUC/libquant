.PHONY: quality

check_dirs := .

style:
	ruff check $(check_dirs) --fix
	ruff format $(check_dirs)

test:
	pytest -v ./tests/ --ignore=tests/accuracy --ignore=tests/performance

test_accuracy:
	pytest -s -v ./tests/accuracy

test_performance:
	pytest -s -v ./tests/performance
