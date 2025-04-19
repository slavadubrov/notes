---
title: Setting up a MacBook as AI Specialist
date: 2025-04-19
tags: [Tutorial, Mac]
summary: A 10-step guide to configuring a new MacBook for AI development, covering essential tools and terminal setup.
---

# Setting up a MacBook as AI Specialist

I believe this story is familiar to everyone. You get a new MacBook and try to recall what exactly do you need there. This happened to me recently. So, I decided to summarize in 10 simple steps what I usually do with my new MacBook. Below you can find the subjective answer to the question, what Data Scientists need to install, and what do they need to configure to simplify their daily routines.

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

4.  I prefer to use [iTerm2](https://www.iterm2.com) instead of the standard Terminal. iTerm provides a more flexible configuration. You can download it from the iTerm2 website.

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

    - [pyenv](https://github.com/pyenv/pyenv) as Python Version Manager,
    - [pipenv](https://pipenv.pypa.io) for Python-Dev Workflow,
    - [htop](https://htop.dev) for process monitoring,
    - [gitmoji](https://gitmoji.dev) for customizing my commit messages.

10. And of course, I always install [VSCode](https://code.visualstudio.com), [Docker](https://www.docker.com), [Nteract](https://nteract.io), and [Postman](https://www.postman.com).

If you read up to this moment:

- you know what I usually use in my daily data science routine;
- it is a pretty simple basic configuration that could help you with your new system configuration.
