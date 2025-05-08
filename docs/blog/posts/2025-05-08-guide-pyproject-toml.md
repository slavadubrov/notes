---
title: Quick-Guide on `pyproject.toml`
date: 2025-05-08
tags: [python, guide]
summary: A practical, modern guide to using pyproject.toml for Python projects—covering packaging, dependencies, tool configuration, and workflows in one file.
---

# Quick-Guide on `pyproject.toml`

## TL;DR

Think of `pyproject.toml` as the **`package.json` for Python**. Whether you prefer `.venv`, `pyenv`, or `uv`, putting all your project's metadata, dependencies, and tooling into one tidy TOML file simplifies development and boosts collaboration.

<!-- more -->

## 1. What **is** `pyproject.toml`?

`pyproject.toml` is a single, [PEP-backed](https://peps.python.org/pep-0518/) configuration file (TOML-format) that sits at the root of your repository.

- [PEP 518](https://peps.python.org/pep-0518/) introduced it so that _build tools_ could declare their requirements in a standard way via a `[build-system]` table.
- [PEP 621](https://peps.python.org/pep-0621/) later standardised a `[project]` table for core package metadata (name, version, dependencies, etc.).

Beyond packaging, many developer tools (Black, isort, pytest, Ruff, etc.) now read their settings from dedicated `[tool.*]` sections, so the file has become a universal "one-stop" project manifest.

## 2. Why should you care?

- **One file to rule them all** - no more juggling `setup.py`, `setup.cfg`, `requirements.txt`, `MANIFEST.in`, and dotfiles.
- **Backend-agnostic builds** - pip reads `pyproject.toml` and installs required build tools automatically.
- **Tooling ecosystem** - linters, formatters, test runners, and type checkers agree on where to look for config.

## 3. Does it _replace_ `requirements.txt`?

Mostly, yes. Modern tools like [Poetry](https://python-poetry.org/), [PDM](https://pdm-project.org/), [Hatch](https://hatch.pypa.io/), and [uv](https://github.com/astral-sh/uv) store dependencies in `[project]` and use lockfiles for reproducibility. `requirements.txt` is only needed for legacy tools or simple CI.

## 4. Anatomy at a glance

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "awesome-app"
version = "0.1.0"
description = "Short demo of pyproject.toml"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
  "fastapi>=0.111",
  "uvicorn[standard]>=0.30",
]

[project.optional-dependencies]
dev = ["pytest", "black", "ruff"]

[tool.black]
line-length = 100
target-version = ["py312"]

[tool.isort]
profile = "black"
```

## 5. Virtual environments & `pyproject.toml`

`pyproject.toml` **does not create a venv itself**; it declares what to install. How you create/activate an environment is up to you:

| Tool         | Command                       | Uses `pyproject.toml`?      |
| ------------ | ----------------------------- | --------------------------- |
| Plain Python | `python -m venv .venv`        | Manual install with pip     |
| pyenv        | `pyenv virtualenv 3.12.2 env` | Yes, if activated correctly |
| uv           | `uv venv` or `uv add ...`     | Yes, automatic + fast       |

## 6. Popular real-world uses

| Use             | Section                     | Tools                                   | Why people like it                                               |
| --------------- | --------------------------- | --------------------------------------- | ---------------------------------------------------------------- |
| Packaging       | `[project]`                 | build, twine, uv                        | Build once with python -m build, upload with twine or uv publish |
| Dependencies    | `[project]` + lock          | Poetry, PDM, uv                         | Reproducible installs; uv sync is very fast                      |
| Formatting      | `[tool.black]`              | [Black](https://black.readthedocs.io/)  | Keeps formatter settings in-repo                                 |
| Sorting imports | `[tool.isort]`              | [isort](https://pycqa.github.io/isort/) | One config shared between IDE and CI                             |
| Testing         | `[tool.pytest.ini_options]` | [pytest](https://docs.pytest.org/)      | No more pytest.ini                                               |
| Typing          | `[tool.mypy]`               | [mypy](https://mypy-lang.org/)          | Optional if you prefer one file over mypy.ini                    |

## 7. Typical workflow

**New project with `uv`:**

```bash
uv init my_app          # scaffolds folder, pyproject.toml and .venv
cd my_app
uv add requests fastapi # adds deps to [project] and installs
uv run pytest           # runs inside the same venv
uv build                # builds wheel/sdist using build-system table
```

**Existing project:**

1. Add a minimal [build-system] to declare setuptools>=61 and wheel.
2. Move metadata and runtime deps into [project] (PEP 621).
3. Convert dev-deps to [project.optional-dependencies].dev.
4. Drop requirements.txt or generate it from the lock file for legacy CI.
5. Configure tools under [tool.*].

## 8. Cheat-sheet

| Task                           | Snippet                                                                                   |
| ------------------------------ | ----------------------------------------------------------------------------------------- |
| Use Flit instead of Setuptools | `[build-system]`<br>`requires=["flit_core>=3.2"]`<br>`build-backend="flit_core.buildapi"` |
| Pin Python version             | `[project]`<br>`requires-python=">=3.12"`                                                 |
| Enable Black formatting        | `[tool.black]`<br>`line-length=88`                                                        |
| Add test dependencies          | `[project.optional-dependencies]`<br>`test=["pytest","coverage"]`                         |
| Export lockfile requirements   | `uv export > requirements.txt`                                                            |

---
