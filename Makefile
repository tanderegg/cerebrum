test:
	python -m unittest discover . "*_tests.py"

mnist:
	python -m examples.mnist.main
