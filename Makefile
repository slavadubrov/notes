.PHONY: install serve build clean setup help

install:
	bundle install

serve:
	bundle exec jekyll serve

build:
	bundle exec jekyll build

clean:
	rm -rf _site/

setup: install

# Help command
help:
	@echo "Available commands:"
	@echo "  make install  - Install dependencies from Gemfile"
	@echo "  make serve    - Start the Jekyll development server"
	@echo "  make build    - Build the static site with Jekyll"
	@echo "  make clean    - Remove the built site directory"
	@echo "  make setup    - Install dependencies" 