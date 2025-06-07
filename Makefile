.PHONY: install serve build clean venv clean-venv check-python check-requirements dev check-updates

venv:
	@if [ ! -d ".venv" ]; then \
		python3 -m venv .venv; \
		source .venv/bin/activate && pip install --upgrade pip; \
		echo "Virtual environment created. Run 'source .venv/bin/activate' to activate it"; \
	else \
		echo "Virtual environment already exists"; \
	fi

clean-venv:
	rm -rf .venv

install: venv
	source .venv/bin/activate && pip install -r requirements.txt

serve: install
	source .venv/bin/activate && mkdocs serve

dev: install
	source .venv/bin/activate && mkdocs serve -a 0.0.0.0:8000 --livereload

build: install
	source .venv/bin/activate && mkdocs build

clean:
	rm -rf site/

setup: clean-venv venv install

help:
	@echo "Available commands:"
	@echo "  make venv           - Create a new virtual environment"
	@echo "  make clean-venv     - Remove the virtual environment"
	@echo "  make install        - Install dependencies from requirements.txt"
	@echo "  make serve          - Start the development server"
	@echo "  make dev            - Start development server with live reload"
	@echo "  make build          - Build the static site"
	@echo "  make clean          - Remove the built site directory"
	@echo "  make setup          - Create new venv and install dependencies"
