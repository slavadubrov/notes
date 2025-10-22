.PHONY: install serve build clean venv clean-venv dev setup sync help

UV ?= uv
VENVDIR ?= .venv

venv:
	$(UV) venv $(VENVDIR)

clean-venv:
	rm -rf $(VENVDIR)

install sync:
	$(UV) sync

serve: install
	$(UV) run mkdocs serve

dev: install
	$(UV) run mkdocs serve -a 0.0.0.0:8000 --livereload

build: install
	$(UV) run mkdocs build

clean:
	rm -rf site/

setup: clean-venv install

help:
	@echo "Available commands:"
	@echo "  make venv           - Create a uv-managed virtual environment"
	@echo "  make clean-venv     - Remove the virtual environment"
	@echo "  make install        - Install dependencies from pyproject.toml via uv"
	@echo "  make serve          - Start the development server"
	@echo "  make dev            - Start development server with live reload"
	@echo "  make build          - Build the static site"
	@echo "  make clean          - Remove the built site directory"
	@echo "  make setup          - Recreate the virtual environment and install dependencies"
