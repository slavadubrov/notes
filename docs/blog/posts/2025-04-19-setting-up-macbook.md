---
title: "Quick-Guide on setting up a MacBook for AI Engineering"
date:
    created: 2025-04-19
    updated: 2025-11-04
tags: [macos, tooling, guide]
categories: [Tooling]
description: A 10-step guide to configuring a new MacBook for AI development, covering essential tools and terminal setup.
author: Viacheslav Dubrov
---

# Quick-Guide on setting up a MacBook for AI Engineering

Setting up a new MacBook for AI development doesn't have to be overwhelming. Here's my streamlined 10-step process to transform a fresh macOS installation into a fully functional AI engineering workstation.

<!-- more -->

## 1. Install Xcode Command Line Tools

Start by installing the Xcode Command Line Tools. These are essential building blocks for any software development on macOS, including AI and data science work.

```bash
xcode-select --install
```

This command opens a dialog that walks you through the installation process.

## 2. Install Homebrew

Next, install [Homebrew](https://brew.sh), the go-to package manager for macOS. It makes installing and managing software incredibly simple. Run this command:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

The installer will guide you through the process and may ask for your password.

## 3. Install essential development tools

Now let's install the core tools you'll need for AI engineering:

```bash
brew install openssl readline sqlite3 xz zlib pyenv uv htop gitmoji pandoc ncdu tmux
```

Here's what each tool does:

**Python environment:**

- [pyenv](https://github.com/pyenv/pyenv) — manage multiple Python versions seamlessly
- [uv](https://github.com/astral-sh/uv) — fast Python package manager and environment handler

**System libraries:**

- [openssl](https://www.openssl.org) — SSL/TLS cryptography support
- [readline](https://tiswww.case.edu/php/chet/readline/rltop.html) — command-line text editing
- [sqlite3](https://sqlite.org) — lightweight embedded database
- [xz](https://tukaani.org/xz) — advanced data compression
- [zlib](https://zlib.net) — compression library

**Productivity tools:**

- [htop](https://htop.dev) — visual system monitor and process viewer
- [tmux](https://github.com/tmux/tmux) — manage multiple terminal sessions
- [ncdu](https://dev.yorhel.nl/ncdu) — analyze disk usage interactively
- [gitmoji](https://gitmoji.dev) — add emojis to commit messages
- [pandoc](https://pandoc.org) — convert documents between formats

> **Note:** For more detailed information about using `uv` for Python development, check out my [Quick-Guide on managing Python on macOS with uv](2025-04-17-uv-on-macos.md).

## 4. Choose your terminal

The default macOS Terminal works fine, but I've found better alternatives. I recently switched from [iTerm2](https://www.iterm2.com) to [Warp](https://www.warp.dev/). Warp is a modern, Rust-based terminal with built-in AI features that make your workflow smoother.

You can download Warp from their [website](https://www.warp.dev/).

### Optional: iTerm2 configuration

If you prefer the battle-tested iTerm2, here's my recommended setup:

**Enable natural text editing:**

1. Open Preferences → Profiles → Keys → Key Mappings
2. Click the Presets… dropdown
3. Select "Natural Text Editing"

**Choose a color theme:**

1. Browse themes at [iTerm2-Color-Schemes](https://github.com/mbadolato/iTerm2-Color-Schemes)
2. Open Preferences → Profiles → Colors → Color Presets…
3. Click Import and select your downloaded theme

## 5. Set up Zsh with Oh My Zsh

Modern macOS comes with Zsh as the default shell, but we'll enhance it with [Oh My Zsh](https://github.com/robbyrussell/oh-my-zsh), a framework that makes Zsh more powerful and easier to customize:

```bash
brew install zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
```

The Oh My Zsh installer will back up your existing Zsh configuration and set up the new one.

## 6. Add Zsh plugins for superpowers

Plugins make your terminal smarter and more productive. Edit your `~/.zshrc` file to add these plugins:

```bash
plugins=(
    aws bgnotify brew docker docker-compose
    emoji forklift gcloud git history iterm2
    keychain kubectl macos pre-commit
    pyenv pylint python screen themes
    tmux virtualenv vscode
    zsh-autosuggestions zsh-syntax-highlighting
)
```

You can find detailed descriptions of all plugins in the [Oh My Zsh plugins wiki](https://github.com/ohmyzsh/ohmyzsh/wiki/Plugins).

**Extra installation required:**

The last two plugins need separate installation (but it's quick!):

- [zsh-autosuggestions](https://github.com/zsh-users/zsh-autosuggestions) — suggests commands as you type based on your history
- [zsh-syntax-highlighting](https://github.com/zsh-users/zsh-syntax-highlighting) — highlights commands in real-time to catch errors

Follow the installation instructions on their respective GitHub pages.

## 7. Make your terminal beautiful with Powerlevel10k

[Powerlevel10k](https://github.com/romkatv/powerlevel10k) is a gorgeous Zsh theme that displays useful information like your current directory, Git status, Python environment, and more. The best part? It comes with an interactive configuration wizard that walks you through customizing it to your preferences.

Follow the [installation instructions](https://github.com/romkatv/powerlevel10k) on their GitHub page.

### Font setup for other editors

If you use VSCode or other editors with integrated terminals, you'll want to use compatible fonts:

1. Open your editor's settings
2. Search for `terminal.integrated.fontFamily`
3. Set it to `MesloLGS NF` (this font is installed with Powerlevel10k)

For detailed font setup instructions, check the [Powerlevel10k font guide](https://github.com/romkatv/powerlevel10k/blob/master/font.md).

## 8. Pick your code editor and AI assistant

For AI engineering, you'll want both a powerful IDE and AI coding assistants. Here's my setup:

**IDE:**

- [Cursor](https://cursor.sh) — a VSCode fork with native AI pair programming features
- [VSCode](https://code.visualstudio.com) — the industry standard with an enormous extension ecosystem

**AI Assistants:**

- [OpenAI Codex](https://openai.com/codex/) — OpenAI's code generation model for intelligent code completion
- [Claude](https://claude.ai) — Anthropic's AI assistant for complex coding tasks and architecture discussions

**My preference:** I use Cursor as my IDE alongside Codex (or Claude Code) running in parallel.

## 9. Additional developer tools

Round out your setup with these applications:

- [GitHub Desktop](https://github.com/apps/desktop) — visual Git client for managing repositories
- [Docker](https://www.docker.com) — containerization platform (or check out [alternatives](https://spacelift.io/blog/docker-alternatives) like Podman)
- [Ollama](https://ollama.com) or [LM Studio](https://lmstudio.ai) — run large language models locally on your Mac

## 10. You're ready to build

That's it! You now have a complete AI engineering setup that mirrors what I use daily. This configuration removes the friction between having an idea and building with AI models. From here, you can:

- Start new Python projects with `uv`
- Run local LLMs for development and testing
- Manage your code with Git and GitHub
- Work efficiently in a beautiful, customized terminal
