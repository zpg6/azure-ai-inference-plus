.PHONY: help install install-dev test format clean build upload examples

help:
	@echo "Available commands:"
	@echo "  install       Install the package"
	@echo "  install-dev   Install package with development dependencies"
	@echo "  test          Run tests"
	@echo "  format        Format code with black and isort"
	@echo "  clean         Clean build artifacts"
	@echo "  build         Build the package"
	@echo "  upload        Upload to PyPI"
	@echo "  examples      Run example usage"
	@echo "  check         Run tests, format code, and check for errors"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v --cov=azure_ai_inference_plus --cov-report=term-missing


format:
	python -m black azure_ai_inference_plus tests examples
	python -m isort azure_ai_inference_plus tests examples

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload: build
	python -m twine upload dist/*

examples:
	@echo "Running basic usage example..."
	cd examples && python basic_usage.py
	@echo "\nRunning embeddings example..."
	cd examples && python embeddings_example.py
	@echo "\nRunning callbacks example..."
	cd examples && python callbacks_example.py

# Development convenience targets
dev-setup: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify everything works"

check: format test
	@echo "All checks passed!" 