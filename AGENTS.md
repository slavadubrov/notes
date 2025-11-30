# Repository Guidelines

## Project Structure & Module Organization

- `docs/` — source content (Markdown). Blog lives under `docs/blog/`. Use Mermaid fences (```mermaid) for diagrams; JS in `docs/javascripts/`.
- `mkdocs.yml` — MkDocs config (theme, plugins, nav).
- `site/` — build output. Do not edit or commit manually.
- Tooling: `Makefile`, `.pre-commit-config.yaml`, `requirements.txt`, optional `.venv/`.

## Build, Test, and Development Commands

- `make venv` — create local virtualenv.
- `make install` — install MkDocs and plugins from `requirements.txt`.
- `make serve` — run local dev server at `http://127.0.0.1:8000`.
- `make dev` — serve with live reload on all interfaces.
- `make build` — produce static site in `site/`.
- `make clean` — remove `site/`.

Example: `make setup && make serve` to get started.

## Coding Style & Naming Conventions

- Markdown: sentence‑case headings, wrap naturally; prefer lists and short paragraphs.
- YAML (`mkdocs.yml`): 2‑space indentation; keep keys sorted logically.
- Pre-commit: trailing whitespace, EOF fixers, ShellCheck, and Prettier run on staged files. Install with `pre-commit install`.
- Blog posts: place under `docs/blog/posts/` named `YYYY‑MM‑DD-title.md`.

## Testing Guidelines

- Run `make build` before opening a PR; the build fails if blog excerpts are missing (`<!-- more -->`) or config is invalid.
- Preview locally with `make serve` and verify navigation, search, tags, and code highlighting.

## Commit & Pull Request Guidelines

- Commits: short imperative subject (e.g., "add uv article", "fix toc"). Group related edits.
- PRs to `main` only. Include:
    - Summary of changes and rationale
    - Screenshots/GIFs for UI or rendering changes
    - Links to related issues (if any)
- Do not commit `site/`. CI builds and deploys on `main` via GitHub Actions.

## Security & Configuration Tips

- Never commit secrets. Deployment uses `GH_PAT` in repository secrets.
- External links: prefer HTTPS; verify integrity of embeds.
- Large assets: avoid binary blobs; link externally or optimize before adding to `docs/`.
