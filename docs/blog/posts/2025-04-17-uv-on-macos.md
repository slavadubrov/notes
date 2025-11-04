---
title: "Quick Guide: Managing Python on macOS with uv"
date:
  created: 2025-04-17
  updated: 2025-11-04
tags: [tooling, python, guide]
description: Lightning-fast Python installs with Rust-powered uv.
author: Viacheslav Dubrov
---

## Quick Start

```bash
# Install uv
brew install uv
uv python install 3.12

# For new projects
uv init                # create project structure
uv add pandas numpy    # add dependencies
uv run train.py        # run your script

# For existing projects with pyproject.toml
uv sync  # install dependencies
uv run train.py  # run your script

# For existing legacy projects with requirements.txt
uv venv                             # create virtual environment
uv pip install -r requirements.txt  # install dependencies
uv run train.py                     # run your script

# Keep uv updated with brew
brew upgrade uv

# Keep uv updated with uv self update
uv self update 
```

<!-- more -->

## Why uv?

`uv` is a blazing-fast Python tool written in Rust that replaces multiple tools with a single, unified experience. It handles:

- **Package management** - installs dependencies 10-100x faster than pip
- **Python installation** - manages multiple Python versions without conflicts
- **Virtual environments** - creates and manages isolated environments automatically
- **Project management** - modern workflow with `pyproject.toml`
- **Tool execution** - runs formatters and linters in isolated sandboxes

The result? Simpler workflows, faster setups, and consistent environments across your laptop and CI.

---

## Installing uv

The easiest way to install `uv` on macOS is via Homebrew:

```bash
brew install uv
```

`uv` automatically detects your Mac's architecture (Apple Silicon or Intel), so no extra configuration is needed.

**Verify the installation:**

```bash
uv --version  # should show 0.6.x or higher
```

**Keep it updated:**

```bash
brew upgrade uv
```

> **Alternative:** If you prefer not to use Homebrew, check the [official installation docs](https://docs.astral.sh/uv/installation) for other options.

---

## Installing Python Versions

`uv` can install and manage multiple Python versions side by side:

```bash
# Install a specific version
uv python install 3.12.4

# Install the latest patch of a version
uv python install 3.13

# Install multiple versions at once
uv python install 3.9 3.10 3.11

# List installed versions
uv python list
```

Python installations are stored in `~/.cache/uv`, keeping them separate from Homebrew or Xcode Python installations.

**Pin a Python version for your project:**

```bash
uv python pin 3.12
```

This creates a `.python-version` file in your project directory. Commit it to Git so your team and CI use the exact same Python version.

---

## Two Ways to Work with uv

### Option 1: Modern Projects (pyproject.toml)

For new projects, `uv` uses the modern `pyproject.toml` standard:

**Start a new project:**

```bash
uv init
```

This creates `pyproject.toml`, `README.md`, and `.gitignore` for you.

**Add dependencies:**

```bash
# Add regular dependencies
uv add pandas numpy

# Add development dependencies (testing, linting, etc.)
uv add pytest --dev

# Add pip (needed for Jupyter notebooks in VS Code)
uv add pip
```

**Run your code:**

```bash
uv run python script.py  # run Python scripts
uv run jupyter lab       # run installed tools
uv run pytest           # run tests
```

**Clone an existing uv project:**

```bash
git clone <repo>
cd <repo>
uv sync  # installs all dependencies with locked versions
```

This ensures everyone on your team uses the exact same package versions.

### Option 2: Traditional Projects (requirements.txt)

For existing projects or when you prefer the traditional approach:

**Set up your environment:**

```bash
# Create virtual environment
uv venv

# Install dependencies
uv pip install -r requirements.txt
```

**Run your code:**

```bash
uv run python script.py
```

> **Tip:** `uv` automatically detects the `.venv` directory, so you don't need to manually activate it.

### Running Tools with uvx

Use `uvx` to run tools without installing them in your project:

```bash
uvx black .          # format code
uvx ruff check src/  # lint code
uvx jupyter lab      # run Jupyter temporarily
```

Each tool runs in its own isolated environment, keeping your project dependencies clean.

---

## Quick Reference

| Task                 | Modern (pyproject.toml)       | Traditional (requirements.txt)       |
| -------------------- | ----------------------------- | ------------------------------------ |
| Start new project    | `uv init`                     | `touch requirements.txt`             |
| Add dependency       | `uv add package`              | `echo package >> requirements.txt`   |
| Install dependencies | `uv sync`                     | `uv pip install -r requirements.txt` |
| Run script           | `uv run python script.py`     | `uv run python script.py`            |
| Run installed tool   | `uv run tool-name`            | `uv run tool-name`                   |
| Create environment   | automatic with `uv init`      | `uv venv`                            |

---

## Using uv with pyenv

You can use both `uv` and `pyenv` together if needed:

- **Keep pyenv** if you need its shell shim strategy to globally control which `python` command is used
- **Use uv alone** if you prefer project-specific Python versions (simpler and faster)

`uv` can use any Python interpreter in your `$PATH`, including those installed by pyenv or Homebrew. Just pass `--python <path>` when needed.

---

## Machine Learning Tips

**PyTorch with CUDA:**
Check out the [PyTorch integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/) for installing GPU-optimized versions.

**Fast experimentation:**
`uv` caches all downloaded packages, so switching between different versions of scikit-learn or TensorFlow is nearly instant.

**Jupyter in VS Code:**
Add pip to your project with `uv add pip` to ensure Jupyter notebooks work properly in VS Code.

---
