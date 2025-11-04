---
title: "Quick-Guide on setting up a MacBook for AI Engineering"
date:
  created: 2025-04-19
  updated: 2025-04-19
tags: [macos, tooling, guide]
description: A 10-step guide to configuring a new MacBook for AI development, covering essential tools and terminal setup.
author: Viacheslav Dubrov
---

# Quick-Guide on setting up a MacBook for AI Engineering

Here's my distilled, 10‑step workflow to transform a vanilla macOS install into a ready to-go AI engineering working station.

<!-- more -->

## 1. Xcode Command Line Tools

First of all, let's install Xcode Command Line Tools. These tools are the foundation for any type of software development (including DS).

```bash
xcode-select --install
```

## 2. Homebrew

Then install [Homebrew](https://brew.sh). It is a package manager for macOS. You can follow instructions on their [website](https://brew.sh) or just run:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

## 3. Main dependencies with brew

Then I install these dependencies with brew:

```bash
brew install openssl readline sqlite3 xz zlib pyenv uv htop gitmoji pandoc yt-dlp ncdu tmux
```

Descriptions:

- [openssl](https://www.openssl.org) for SSL/TLS cryptography,
- [readline](https://tiswww.case.edu/php/chet/readline/rltop.html) for command-line text editing support,
- [sqlite3](https://sqlite.org) for embedded SQL database engine,
- [xz](https://tukaani.org/xz) for LZMA2-based data compression,
- [zlib](https://zlib.net) for DEFLATE compression library,
- [pyenv](https://github.com/pyenv/pyenv) for managing global Python versions via shims,
- [uv](https://github.com/astral-sh/uv) for managing Python project dependencies and virtual environments,
- [htop](https://htop.dev) for interactive process viewer and system monitor,
- [gitmoji](https://gitmoji.dev) for customizing commit messages.
- [pandoc](https://pandoc.org) for converting documents between various markup formats,
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for downloading videos from online platforms,
- [ncdu](https://dev.yorhel.nl/ncdu) for analyzing disk usage in a terminal interface,
- [tmux](https://github.com/tmux/tmux) for terminal session multiplexing.

> For more detailed information about using `uv` for Python development, check out my [Quick-Guide on managing Python on macOS with uv](2025-04-17-uv-on-macos.md).

## 4. Terminals

I used to prefer [iTerm2](https://www.iterm2.com) over the standard Terminal due to its flexible configuration, but recently migrated to [Warp](https://www.warp.dev/). Warp offers a modern, Rust-based terminal experience with AI features integrated. You can download it from the Warp website. However, if you still prefer iTerm2, here's how I used to configure it:

## 5. iTerm configuration

For configuring iTerm I prefer to do the following:

- Setup Natural text editing:
    - Go to Preferences → Profiles → Keys → Key Mappings
    - Press Presets… dropdown button
    - Select Natural Text Editing

- For changing color select the preferred preset from this [repo](https://github.com/mbadolato/iTerm2-Color-Schemes). Then:
    - Go to Preferences → Profiles → Colors → Color Presets… → Import (or select)
    - After importing your new color will be displayed in Color Presets

## 6. Zsh + Oh My Zsh

Then I install and configure [Zsh](https://www.zsh.org) and [Oh My Zsh](https://github.com/robbyrussell/oh-my-zsh):

```bash
brew install zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
```

## 7. Zsh plugins

Now you can configure your terminal with a `~/.zshrc` file. I use the next zsh plugins in my daily routine:

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

A description of the plugins you can find [here](https://github.com/ohmyzsh/ohmyzsh/wiki/Plugins).

Only the last two plugins (zsh-autosuggestions and zsh-syntax-highlighting) require additional installation. It's pretty simple, just check the following links:

- [zsh-autosuggestions](https://github.com/zsh-users/zsh-autosuggestions)
- [zsh-syntax-highlighting](https://github.com/zsh-users/zsh-syntax-highlighting)

## 8. Powerlevel10k

I'm using the [Powerlevel10k](https://github.com/romkatv/powerlevel10k) theme in Zsh. It has installation assistance that helps you configure Zsh your way. Just follow the instruction on their [website](https://github.com/romkatv/powerlevel10k).

If you have any issues with fonts in another terminal, you can install fonts separately.
For example, to configure VSCode to use the Nerd Font either follow this [instruction](https://github.com/romkatv/powerlevel10k/blob/master/font.md) or do the next:

1. Open VSCode Settings:
2. Set the Terminal Font:
      1. Search for terminal.integrated.fontFamily.
      2. Set its value to the name of the installed font, e.g., `MesloLGS NF`.

## 9. IDE

I prefer [VSCode](https://code.visualstudio.com) or [Cursor](https://cursor.sh).

## 10. Other developer tools

- [GitHub Desctop](https://github.com/apps/desktop)
- [Docker](https://www.docker.com) or [its alternatives](https://spacelift.io/blog/docker-alternatives)
- Local LLMs clients like [Ollama](https://ollama.com) or [LMStudio](https://lmstudio.ai)

## Final thoughts

By now you have the exact stack I lean on every day as an AI engineer - just the essentials that remove friction between an idea and a running model.
