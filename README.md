# Edge of Context: ML Blog

A blog about Machine Learning tips, tricks, and experiences, built with MkDocs and hosted on GitHub Pages.

## 🛠 Tech Stack

- [MkDocs](https://www.mkdocs.org/) - Static site generator
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) - Beautiful theme
- GitHub Pages - Hosting
- GitHub Actions - Automated deployment

## 🚀 Quick Start

1. Clone the repository:

    ```bash
    git clone <your-repo-url>
    cd <repo-name>
    ```

2. Ensure you have [uv](https://github.com/astral-sh/uv) installed (see install instructions in the uv docs).

3. Install dependencies and launch a preview:

    ```bash
    uv sync                  # Installs runtime + dev dependencies into .venv
    uv run mkdocs serve      # open http://127.0.0.1:8000
    ```

4. Alternatively, start the development server via Make:

    ```bash
    make serve
    ```

## 📝 Creating Content

Blog posts are written in Markdown and stored in the `docs/` directory. To create a new post:

1. Create a new `.md` file in the `docs/` directory
2. Add the file to the navigation in `mkdocs.yml`
3. Write your content using Markdown

## 🛠 Available Commands

- `make help` - Show all available commands
- `make setup` - Recreate the uv environment from scratch
- `make install` - Install dependencies declared in `pyproject.toml`
- `make serve` - Start the development server
- `make build` - Build the static site
- `make clean` - Remove the built site
- `make venv` - Create a new virtual environment
- `make clean-venv` - Remove the virtual environment
- `make svg-to-png` - Convert SVG diagrams to PNG (incremental)
- `make svg-to-png-force` - Regenerate all PNG files from SVGs
- `make test-examples FILE=path/to/post.md` - Test Python code blocks in markdown

## 🖼 Generating PNG Diagrams for Substack

SVG diagrams are stored in `docs/blog/assets/` and need to be converted to PNG for platforms like Substack that don't support SVG. PNG files are **not committed to git** — they must be generated locally.

### First-time setup

Install Playwright's Chromium browser (~160MB download, one-time only):

```bash
uv run --with playwright playwright install chromium
```

### Generate PNGs

```bash
# Convert only new/missing PNGs (fast, incremental)
make svg-to-png

# Force regenerate all PNGs
make svg-to-png-force
```

PNGs are saved next to their source SVG files at 2x resolution for retina quality.

## 🧪 Testing Python Code Examples

Validate that Python code blocks in blog posts actually work. The script extracts every ` ```python ` fenced code block from a markdown file and runs each as an isolated test case.

```bash
make test-examples FILE=docs/blog/posts/2025-12-25-schema-guided-reasoning-vllm.md
```

This helps catch:

- Syntax errors in code snippets
- Broken imports or missing dependencies
- Examples that no longer work after API changes

## 📦 Dependencies

- Runtime and development dependencies are declared in [`pyproject.toml`](pyproject.toml).
- Use `uv add <package>` and `uv add --dev <package>` to manage them.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
