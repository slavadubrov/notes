---
title: Quick-Guide on ~/.zprofile vs ~/.zshrc 🚀
date: 2025-05-07
tags: [tooling, macos, guide]
summary: A tiny guide explaining the differences between ~/.zprofile and ~/.zshrc, their use cases, and best practices for shell configuration.
---

# Quick-Guide on ~/.zprofile vs ~/.zshrc 🚀

## TL;DR ⚡

- **`~/.zprofile`** → one-shot, login-shell initialization (think "environment/bootstrap") 🔧
- **`~/.zshrc`** → every interactive prompt (think "daily driving experience") 🎮

Use both in tandem: keep your environment reliable with **`~/.zprofile`**, and your shell pleasant and tweakable with **`~/.zshrc`** ✨

<!-- more -->

## 1. Two Kinds of Z-shells 🐚

| Shell starts as…      | Technically called | Typical triggers                                                      | Reads first   | Purpose                        |
| :-------------------- | :----------------- | :-------------------------------------------------------------------- | :------------ | :----------------------------- |
| **Login shell**       | _login_            | • New terminal tab/window on macOS<br>• `ssh user@host`<br>• `zsh -l` | `~/.zprofile` | One-time, "session-wide" setup |
| **Interactive shell** | _interactive_      | • Every prompt you see after the shell is running                     | `~/.zshrc`    | Live, inter-prompt behavior    |

> **Note:** A login shell is also interactive, but only login shells run the "login-only" startup files.

---

## 2. What Goes Where 📁

| Put in `~/.zprofile` (login-only) 🔐                                                                                             | Put in `~/.zshrc` (every prompt) 🔄                                       |
| :------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------ |
| • `export PATH=…` and other environment variables that downstream programs need **before** they launch (e.g., GUI apps on macOS) | • Aliases, functions, and shell options (`setopt autocd`)                 |
| • `eval "$(pyenv init -)"`, NVM, ASDF, etc.—tools that must adjust `$PATH` early                                                 | • Prompt theming (`POWERLEVEL9K_*`), key-bindings, completion tweaks      |
| • `ulimit`, `ssh-agent` start-up, `launchctl` tweaks                                                                             | • History settings, auto-suggest-highlight, `bindkey` mappings            |
| • Anything you want to run **once** per terminal/SSH login                                                                       | • Anything you want to re-load with `source ~/.zshrc` without logging out |

---

## 3. Practical Rules of Thumb 📝

1. **Session vs. Prompt** ⏱️
   If it should happen once per session, choose **`~/.zprofile`**; if it should affect every new prompt, choose **`~/.zshrc`**.

2. **PATH-mangling early?** 🛣️
   Put it in `~/.zprofile` so every child process inherits it.

3. **Re-sourcing convenience?** 🔄
   Keep interactive tweaks in `~/.zshrc`; you can iterate without closing the terminal.

4. **Platform nuance** 💻

   - macOS Terminal & iTerm2 open **login** shells by default ⇒ both files run.
   - Linux desktop terminals (GNOME, Kitty, Alacritty) usually start **non-login** shells ⇒ only `~/.zshrc` runs; add `source ~/.zprofile` to `~/.zshrc` _if_ you need login code there.

5. **Remote scripts** (`#!/usr/bin/env zsh -l`)—use a login shell if you rely on `~/.zprofile` 🌐

---

## 4. Minimal Template 📋

```zsh
# ~/.zprofile  --------------------------------
# Environment & once-per-session setup
export PATH="$HOME/.local/bin:$PATH"
eval "$(fnm env)"        # Node version manager

# ~/.zshrc  -----------------------------------
# Prompt, aliases, keybindings
autoload -Uz promptinit; promptinit
prompt pure
alias gs='git status'
bindkey -e
```
