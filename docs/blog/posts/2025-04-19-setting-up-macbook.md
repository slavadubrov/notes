---
title: Setting up a MacBook for AI Engineering
date: 2025-04-19
tags: [Tutorial, Mac]
summary: A 10-step guide to configuring a new MacBook for AI development, covering essential tools and terminal setup.
---

# Setting up a MacBook for AI Engineering

Here’s my distilled, 10‑step workflow to transform a vanilla macOS install into a ready to-go AI engineering working station.

<!-- more -->

1.  First of all, let's install Xcode Command Line Tools. These tools are the foundation for any type of software development (including DS).

    ```bash
    xcode-select --install
    ```

2.  Then install [Homebrew](https://brew.sh). It is a package manager for macOS. You can follow instructions on their [website](https://brew.sh) or just run:

    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

3.  Then I install these dependencies with brew:

    ```bash
    brew install openssl readline sqlite3 xz zlib
    ```

4.  I used to prefer [iTerm2](https://www.iterm2.com) over the standard Terminal due to its flexible configuration, but recently migrated to [Warp](https://www.warp.dev/). Warp offers a modern, Rust-based terminal experience with AI features integrated. You can download it from the Warp website. However, if you still prefer iTerm2, here's how I used to configure it:

5.  For configuring iTerm I prefer to do the following:

    - Setup Natural text editing:

      - Go to Preferences → Profiles → Keys → Key Mappings
      - Press Presets… dropdown button
      - Select Natural Text Editing

    - For changing color select the preferred preset from this [repo](https://github.com/mbadolato/iTerm2-Color-Schemes). Then:
      - Go to Preferences → Profiles → Colors → Color Presets… → Import
      - After importing your new color will be displayed in Color Presets

6.  Then I install and configure [Zsh](https://www.zsh.org) and [Oh My Zsh](https://github.com/robbyrussell/oh-my-zsh):

    ```bash
    brew install zsh
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
    ```

7.  Now you can configure your terminal with a `~/.zshrc` file. I use the next zsh plugins in my daily routine:

    ```bash
    plugins=(
      git brew vscode iterm2
      themes screen macos bgnotify
      docker docker-compose gcloud aws
      python pyenv pylint virtualenv
      zsh-autosuggestions zsh-syntax-highlighting)
    ```

    A description of the plugins you can find [here](https://github.com/ohmyzsh/ohmyzsh/wiki/Plugins).

    Only the last two plugins (zsh-autosuggestions and zsh-syntax-highlighting) require additional installation. It's pretty simple, just check the following links:

    - zsh-autosuggestions: https://github.com/zsh-users/zsh-autosuggestions
    - zsh-syntax-highlighting: https://github.com/zsh-users/zsh-syntax-highlighting/

8.  I'm using the [Powerlevel10k](https://github.com/romkatv/powerlevel10k) theme in Zsh. It has installation assistance that helps you configure Zsh your way. Just follow the instruction on their [website](https://github.com/romkatv/powerlevel10k).

    If you have any issues with fonts in another terminal, you can install fonts separately. For example, I use VSCode and its internal terminal very often. This [instruction](https://github.com/romkatv/powerlevel10k/blob/master/font.md) could help you configure the VSCode terminal to work with the Powerlevel10k theme.

9.  For the terminal I also install:

    - [pyenv](https://github.com/pyenv/pyenv) for managing global Python versions via shims,
    - [uv](https://github.com/astral-sh/uv) for managing Python project dependencies and virtual environments,
    - [htop](https://htop.dev) for process monitoring,
    - [gitmoji](https://gitmoji.dev) for customizing commit messages.

10. And of course, I always install [Cursor](https://cursor.sh) as my main IDE, [Docker](https://www.docker.com) for containerization, and [Ollama](https://ollama.com) for running LLMs locally.

## Final thoughts

By now you have the exact stack I lean on every day as an AI engineer - just the essentials that remove friction between an idea and a running model.
