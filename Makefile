.PHONY: install serve build clean venv clean-venv

venv:
	python3 -m venv .venv
	@echo "Run 'source .venv/bin/activate' to activate the virtual environment"

clean-venv:
	rm -rf .venv

install:
	pip install -r requirements.txt

serve:
	mkdocs serve

build:
	mkdocs build

clean:
	rm -rf site/

setup: clean-venv venv install

# Help command
help:
	@echo "Available commands:"
	@echo "  make venv     - Create a new virtual environment"
	@echo "  make clean-venv - Remove the virtual environment"
	@echo "  make install  - Install dependencies from requirements.txt"
	@echo "  make serve    - Start the development server"
	@echo "  make build    - Build the static site"
	@echo "  make clean    - Remove the built site directory"
	@echo "  make setup    - Create new venv and install dependencies"
