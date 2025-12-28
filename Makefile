.PHONY: install serve build clean venv clean-venv dev setup sync svg-to-png svg-to-png-force test-examples help

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

svg-to-png:
	$(UV) run --with playwright python scripts/svg_to_png.py

svg-to-png-force:
	$(UV) run --with playwright python scripts/svg_to_png.py --force

test-examples:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make test-examples FILE=path/to/post.md"; \
		exit 1; \
	fi
	$(UV) run python scripts/run_markdown_examples.py $(FILE)

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
	@echo "  make svg-to-png     - Convert SVGs to PNG (only missing)"
	@echo "  make svg-to-png-force - Regenerate all PNG files from SVGs"
	@echo "  make test-examples FILE=path/to/post.md - Test Python code blocks in a markdown file"
