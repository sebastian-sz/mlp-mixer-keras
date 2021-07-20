lint:
	pre-commit run --all-files

test:
	python -m unittest tests/*.py
