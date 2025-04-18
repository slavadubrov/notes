---
title: Managing Python on macOS with uv
date: 2025-04-17
tags: [Tooling, MLE, Python, macOS, "Virtual Environments", CLI, Tutorial]
summary: Lightning-fast Python installs with Rust-powered uv.
---

# Managing Python like an ML Engineer on macOS with **uv**

## TL;DR Bash Cheat‑sheet

```bash
brew install uv        # install tool
uv python install 3.12 # grab interpreter
uv python pin          # lock version for repo
uv venv                # create .venv
uv pip install numpy pandas   # ML staples
uv run train.py        # run with correct interpreter
uv self upgrade        # update uv itself
```

---

## 🌙 Why I Migrated to `uv` (And You Should Too)

`uv` is a lightning-fast, all-in-one Python project tool written in Rust, combining package management, interpreter installation, and virtual environment creation. Key features include:

- Installing and switching between multiple CPython (and PyPy) builds
- Creating lightweight virtual environments
- Resolving dependencies with an absurdly fast pip-compatible resolver
- A `uvx` shim for running tools like Ruff or Black in isolated sandboxes:
  - `uvx black .` or `uvx ruff format .`

Result: fewer moving parts, faster setups, and consistent environments across laptop and CI images.

---

## 1. Installing uv

```bash
# Install uv via Homebrew (Apple Silicon & Intel)
brew install uv
```

> **Note:** `uv` auto-detects your architecture (Apple Silicon or Intel).

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

## 3. Virtual environments the lazy way

```bash
# Create virtual environment
uv venv                 # creates .venv with the pinned Python
uv venv --python 3.11   # override if you're exploring
```

I rarely `activate` anymore as uv detects the `.venv` file and routes `uv pip`, `uv run`, or `uvx ruff` to the right interpreter. Pure convenience.

A few patterns that shaved minutes from my workflow:

| Task                                      | One‑liner                               |
| ----------------------------------------- | --------------------------------------- |
| Install deps into the current venv        | `uv pip install -r requirements.txt`    |
| Run a script with a different interpreter | `uv run --python 3.10 scripts/train.py` |
| Global tool in its own sandbox            | `uvx ruff format .`                     |

### 3.1 Using `uvx` for tools

With `uvx` you can run formatters or linters without touching your virtual environment:

```bash
uvx black .            # format code in an isolated sandbox
uvx ruff check src/    # lint code without installing ruff globally
```

---

## 4. Co‑existing with pyenv (if you must)

- **Keep pyenv** if you rely on its "shim" strategy to globally shadow `python` in your shell.
- **Skip pyenv** if project‑local versions and CI parity are your priority - uv handles that solo.

From uv's perspective every interpreter in `$PATH` (even ones compiled by pyenv or Homebrew) is just "system Python". You can pass it to any `--python` flag and mix‑and‑match as needed.

---

## 5. ML‑specific niceties

- The **[PyTorch integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/)** shows CUDA‑aware installs in one command - excellent for GPU vs. CPU builds on the same Mac.
- Binary wheels pulled by uv are cached, so re‑creating a venv to try a different version of scikit‑learn or TensorFlow feels instant.

---
