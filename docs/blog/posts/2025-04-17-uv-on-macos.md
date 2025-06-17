---
title: Quick-Guide on managing Python on macOS with uv
date: 2025-04-17
tags: [tooling, python, guide]
summary: Lightning-fast Python installs with Rust-powered uv.
---

# Quick-Guide on managing Python like an AI Engineer on macOS with **uv**

## TL;DR Bash Cheat‑sheet

```bash
brew install uv        # install tool
uv python install 3.12 # grab interpreter

# New project workflow (modern)
uv init                # create new project with pyproject.toml
uv add pandas numpy    # add dependencies
uv run train.py        # run with correct interpreter

# Classical project workflow (requirements.txt)
uv venv                           # create .venv
uv pip install -r requirements.txt # install from requirements
uv run train.py                   # run script

uv self upgrade        # update uv itself
```

---

<!-- more -->

## 🌙 Why I Migrated to `uv` (And You Should Too)

`uv` is a lightning-fast, all-in-one Python project tool written in Rust, combining package management, interpreter installation, and virtual environment creation. Key features include:

- Installing and switching between multiple CPython (and PyPy) builds
- Creating lightweight virtual environments
- Resolving dependencies with an absurdly fast pip-compatible resolver
- Modern project management with `pyproject.toml`
- A `uvx` shim for running tools like Ruff or Black in isolated sandboxes:
  - `uvx black .` or `uvx ruff format .`

Result: fewer moving parts, faster setups, and consistent environments across laptop and CI images.

---

## 1. Installing uv

```bash
# Install uv via Homebrew (Apple Silicon & Intel)
brew install uv
```

> **Note:** `uv` auto-detects your architecture (Apple Silicon or Intel).

The same page shows a [one‑liner curl installer](https://docs.astral.sh/uv/installation) if you're brew‑averse.
Check it worked:

```bash
# Check installation
uv --version      # should print something like 0.6.x
uv self upgrade   # keep it fresh
```

---

## 2. Installing Python interpreters

```bash
# Install specific Python versions
uv python install 3.12.4          # exact version
uv python install 3.13            # latest minor
uv python install 3.9 3.10 3.11   # many at once
uv python list                    # what's already cached
```

These archives live under `~/.cache/uv`, so they don't fight Homebrew or Xcode.

Need the interpreter for _this_ project only?

```bash
# Pin Python version for the project
uv python pin           # writes .python-version next to your code
```

Drop that file into Git and your team (or the CI) will automatically get the same binary.

---

## 3. Two Workflows: Modern vs Classical

### 3.1 Modern Workflow: New Projects with `pyproject.toml`

For new projects or when you want to embrace the modern Python packaging ecosystem:

```bash
# Start a new project
uv init                    # creates pyproject.toml, README.md, .gitignore

# Add dependencies
uv add pip                 # needed for Jupyter notebooks in VS Code
uv add pandas numpy        # add your ML packages
uv add pytest --dev       # add development dependencies

# Run your code
uv run python script.py    # run scripts
uv run jupyter lab         # run installed tools
uv run pytest             # run tests
```

**When cloning an existing uv project:**

```bash
git clone <repo>
cd <repo>
uv sync                    # installs all dependencies from pyproject.toml
```

This creates a complete environment with locked dependencies, ensuring reproducible builds across your team.

### 3.2 Classical Workflow: Existing Projects with `requirements.txt`

For existing projects or when working with traditional Python setups:

```bash
# Set up environment
uv venv                           # creates .venv
uv pip install -r requirements.txt # install dependencies

# Run your code
uv run python script.py           # run scripts
uv run installed-package          # run any installed CLI tools
```

> **Pro tip:** `uv` automatically detects the `.venv` directory, so you rarely need to manually activate environments.

### 3.3 Using `uvx` for Global Tools

With `uvx` you can run formatters or linters without touching your virtual environment:

```bash
uvx black .            # format code in an isolated sandbox
uvx ruff check src/    # lint code without installing ruff globally
uvx jupyter lab        # run Jupyter without installing it locally
```

---

## 4. Quick Reference Table

| Task                                      | Modern (`pyproject.toml`)     | Classical (`requirements.txt`)           |
| ----------------------------------------- | ----------------------------- | ---------------------------------------- |
| Start new project                         | `uv init`                     | `touch requirements.txt`                 |
| Add dependency                            | `uv add package`              | `echo package >> requirements.txt`      |
| Install dependencies                      | `uv sync`                     | `uv pip install -r requirements.txt`    |
| Run script                                | `uv run python script.py`     | `uv run python script.py`               |
| Run installed tool                        | `uv run tool-name`            | `uv run tool-name`                       |
| Create environment                        | _automatic with uv init_      | `uv venv`                                |

---

## 5. Co‑existing with pyenv (if you must)

- **Keep pyenv** if you rely on its "shim" strategy to globally shadow `python` in your shell.
- **Skip pyenv** if project‑local versions and CI parity are your priority - uv handles that solo.

From uv's perspective every interpreter in `$PATH` (even ones compiled by pyenv or Homebrew) is just "system Python". You can pass it to any `--python` flag and mix‑and‑match as needed.

---

## 6. ML‑specific niceties

- The **[PyTorch integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/)** shows CUDA‑aware installs in one command - excellent for GPU vs. CPU builds on the same Mac.
- Binary wheels pulled by uv are cached, so re‑creating a venv to try a different version of scikit‑learn or TensorFlow feels instant.
- Use `uv add pip` in new projects to ensure Jupyter notebooks work seamlessly in VS Code.

---
