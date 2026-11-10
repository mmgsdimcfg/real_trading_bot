# Repository layout

- `AutoBot/` - auto-trading code (r003 live trading, scanners/collectors/backtests, control CLI r016). Runs on **both** hosts:
  the Raspberry Pi (Ubuntu, systemd) and the Windows notebook (AutoBotH local mode, r020 supervisor). Its own rules:
  `AutoBot/CLAUDE.md`, `AutoBot/AGENTS.md`.
- `AutoBotH/` - Windows Electron app AutoBotH (display + control through `AutoBot/live_trading/r016_control.py`, over SSH exec for the
  Pi or as a local child process on the notebook).
- `runtime/` and `open-trading-api/` live OUTSIDE the repo, next to the repo folder (`<repo>/../runtime`) on each host.

Rules
1. One KIS account, two hosts: r003 may trade on only one host at a time. The host role file (`runtime/config/host_role.json`,
   `r016 role`) marks the other one standby; r003 refuses to start on a standby host. Never add a start path that skips this
   (AutoBotH `peer.ts` guardStart, r003 `_host_role_refusal`).
2. The app never calls KIS order APIs and never re-implements trading decisions. Every change of bot state goes through `r016_control.py`
   (both hosts) - the app writes no bot files itself, except the one-time KIS config import into the notebook's own `%USERPROFILE%\KIS\config`.
3. AutoBot code must stay runnable on Linux and Windows: file locks / process checks / detached starts go through
   `live_trading/r019_platform.py` (no direct `fcntl`, `/proc`, `systemctl` outside the systemd backend of r016).
4. A change to an r016/r017/r021 JSON contract must update the app (`AutoBotH/`) and the contract docs
   (`AutoBot/autoboth_*.md`, latest: `AutoBot/autoboth_local_mode_20261110.md`) in the same change set.
5. Prefix commit messages with `autobot:` or `autoboth:` (or both).
