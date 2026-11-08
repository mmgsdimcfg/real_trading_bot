# Repository layout

- `AutoBot/` - Ubuntu (Raspberry Pi) auto-trading server code. Its own rules: `AutoBot/CLAUDE.md`, `AutoBot/AGENTS.md`.
- `AutoBotH/` - Windows Electron app AutoBotH (display + control via SSH exec of `AutoBot/live_trading/r016_control.py`).
- `runtime/` and `open-trading-api/` live OUTSIDE the repo, next to the repo folder (`<repo>/../runtime`).

Rules
1. Live trading runs only on the server. The app never calls KIS APIs, never writes server files directly, never re-implements trading decisions.
2. A change to an r016/r017 JSON contract must update the app (`AutoBotH/`) and the contract docs in the same change set.
3. Prefix commit messages with `autobot:` or `autoboth:` (or both).
