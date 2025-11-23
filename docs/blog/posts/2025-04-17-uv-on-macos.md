---
title: "Quick Guide: Managing Python on macOS with uv"
date:
  created: 2025-04-17
  updated: 2025-11-23
tags: [tooling, python, guide]
description: Lightning-fast Python installs with Rust-powered uv.
author: Viacheslav Dubrov
---

## Quick Start

```bash
# Install uv
brew install uv

# For new projects (modern workflow)
uv init                # create project structure
uv add pandas numpy    # add dependencies
uv run train.py        # run your script

# For existing projects (legacy workflow)
uv venv                             # create virtual environment
uv pip install -r requirements.txt  # install dependencies
uv run train.py                     # run your script

# Run tools without installing them
uvx ruff check .       # run linter
uvx black .            # run formatter
```

<!-- more -->

## Why uv?

If you've been using Python for a while, you're likely familiar with the "tool fatigue" of managing `pip`, `virtualenv`, `pip-tools`, `pyenv`, and `poetry`.

**`uv` replaces all of them.**

Written in Rust, it is designed to be a drop-in replacement that is **10-100x faster** than existing tools. It unifies your workflow into a single, cohesive experience.

![The uv Ecosystem](../assets/uv-on-macos/uv-ecosystem.svg)

It handles:

- **Package management** (replacing `pip` and `pip-tools`)
- **Python installation** (replacing `pyenv`)
- **Virtual environments** (replacing `virtualenv` and `venv`)
- **Tool execution** (replacing `pipx`)
- **Project management** (replacing `poetry` or `pdm`)

---

## Installing uv

The easiest way to install `uv` on macOS is via Homebrew:

```bash
brew install uv
```

`uv` automatically detects your Mac's architecture (Apple Silicon or Intel), so no extra configuration is needed.

**Keep it updated:**

```bash
brew upgrade uv
# OR
uv self update
```

---

## Core Concepts

`uv` simplifies Python development by handling three distinct use cases:

1. **Projects**: Building an application or library with dependencies.
2. **Scripts**: Running a single-file Python script with inline dependencies.
3. **Tools**: Running command-line utilities (like `ruff` or `httpie`) globally.

### 1. Modern Project Management

For new projects, `uv` uses the standard `pyproject.toml` for configuration and a cross-platform `uv.lock` for reproducible builds.

![Modern uv Project Structure](../assets/uv-on-macos/uv-project-structure.svg)

**Start a new project:**

```bash
uv init my-project
cd my-project
```

This creates a clean project structure with a `pyproject.toml`, `.gitignore`, and a `hello.py`.

**Add dependencies:**

```bash
# Add runtime dependencies
uv add pandas requests

# Add development dependencies
uv add pytest ruff --dev
```

**Run your code:**

```bash
uv run hello.py
```

`uv` automatically manages the virtual environment in `.venv`. You never need to manually activate it!

### 2. Managing Python Versions

Forget `pyenv`. `uv` can install and manage Python versions for you, keeping them isolated in `~/.cache/uv`.

**Install a specific version:**

```bash
uv python install 3.12
```

**Pin a version for your project:**

```bash
uv python pin 3.11
```

This creates a `.python-version` file. When you run `uv run`, it will automatically use the pinned version, downloading it if necessary. This ensures your entire team and CI pipeline use the _exact same Python version_.

### 3. Running Tools with `uvx`

Use `uvx` (an alias for `uv tool run`) to execute Python command-line tools without polluting your global environment or project dependencies.

```bash
# Run a linter
uvx ruff check .

# Run a formatter
uvx black .

# Start a temporary Jupyter server
uvx --from jupyterlab jupyter lab
```

Each tool runs in its own isolated, temporary environment. It's fast, clean, and safe.

---

## Legacy Projects (requirements.txt)

If you have an existing project using `requirements.txt`, `uv` works as a drop-in replacement for `pip` and `venv`.

**Setup:**

```bash
# Create a virtual environment
uv venv

# Install dependencies (lightning fast!)
uv pip install -r requirements.txt
```

**Run:**

```bash
uv run python app.py
```

---

## Performance Notes

Why is `uv` so fast?

1. **Rust**: It's built with performance in mind, without the overhead of Python startup times.
2. **Global Cache**: It caches built wheels globally. If you've installed `numpy` in one project, installing it in another is instant (using copy-on-write links on macOS).
3. **Parallelism**: It downloads and installs packages in parallel, maximizing your bandwidth.

---

## Summary

| Task                | Old Way                                               | The uv Way               |
| :------------------ | :---------------------------------------------------- | :----------------------- |
| **Install Python**  | `pyenv install 3.12`                                  | `uv python install 3.12` |
| **New Project**     | `mkdir proj && cd proj && python -m venv .venv`       | `uv init proj`           |
| **Install Package** | `pip install pandas && pip freeze > requirements.txt` | `uv add pandas`          |
| **Run Script**      | `source .venv/bin/activate && python script.py`       | `uv run script.py`       |
| **Run Tool**        | `pipx run black`                                      | `uvx black`              |

Switching to `uv` on macOS is one of the highest-ROI changes you can make to your Python workflow today. It's faster, simpler, and standard-compliant.
