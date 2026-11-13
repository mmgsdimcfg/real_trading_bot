# -*- coding: utf-8 -*-
"""R023 bot holdings ledger - which part of each account holding was bought by the bot (r003) and which by hand (MTS/HTS).

Overview
- KIS reports one quantity per symbol; it does not say who bought it. r003 keeps this ledger next to the account sync:
  for every held symbol `acct_qty` (the account quantity at the last sync) and `bot_qty` (the part the bot bought).
  manual quantity = acct_qty - bot_qty.
- The ledger is quantity-only and is driven by the account quantity itself (not by order fills), so partial fills,
  reprices and cancelled remainders cannot make it drift:
    account quantity went UP    -> bot's if the bot had a buy order working for that symbol in this interval, else manual
    account quantity went DOWN  -> taken from the bot part first if the bot had a sell order working, else from the
                                   manual part first (the user sold by hand), the rest from the other part
    bot_qty is always clamped to 0..acct_qty; a symbol that left the account leaves the ledger.
- Files (both under <runtime>/data/live_runtime, written atomically):
    bot_holdings.json  the ledger, kept across days and restarts
    holdings.json      a display snapshot of the whole account for r016 `holdings` (AutoBotH 보유 현황)
- No broker import here: r003 passes plain dicts. Pure functions are unit-tested (tests/test_r023_bot_holdings.py).

Limits (shown to the user in the app / docs)
- A bot buy and a manual buy of the SAME symbol inside the same sync interval while the bot's order is working are both
  counted as the bot's (the account only shows the sum).
- First run (no ledger file yet): seeded from the persisted live state - a holding is the bot's when today's or an earlier
  day's live state shows a bot buy for it (traded_today / a non-"synced" buy session / the re-entry "bot" flag), with its
  whole account quantity; everything else is manual.

Update log:
- [2026-11-13] type=feat owner=claude
    summary: new module (user request: show bot-bought and hand-bought holdings separately, sell only the bot's part).
    impact: live (r003 account sync / place_sell_order, r016 holdings)
    compatibility: backward-compatible (new files; sell behaviour changes only through r001 AUTO_SELL_SCOPE)
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

LEDGER_FILENAME = "bot_holdings.json"
SNAPSHOT_FILENAME = "holdings.json"
SEED_LOOKBACK_FILES = 10


def _norm(code) -> str:
    return str(code).strip().zfill(6)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def normalize_ledger(raw) -> dict[str, dict]:
    """{code: {"acct_qty": int >= 1, "bot_qty": 0..acct_qty}} from anything dict-like (bad rows are dropped)."""
    out: dict[str, dict] = {}
    for code, row in (raw or {}).items():
        if not isinstance(row, dict):
            continue
        try:
            acct = int(row.get("acct_qty", 0) or 0)
            bot = int(row.get("bot_qty", 0) or 0)
        except (TypeError, ValueError):
            continue
        if acct <= 0:
            continue
        out[_norm(code)] = {"acct_qty": acct, "bot_qty": max(0, min(bot, acct))}
    return out


def load_ledger(live_runtime_dir: Path) -> dict[str, dict] | None:
    """the stored ledger, or None when there is no (readable) ledger file yet -> the caller seeds one"""
    raw = _read_json(Path(live_runtime_dir) / LEDGER_FILENAME)
    if raw is None or not isinstance(raw.get("codes"), dict):
        return None
    return normalize_ledger(raw["codes"])


def save_ledger(live_runtime_dir: Path, ledger: dict[str, dict], now: datetime | None = None) -> None:
    _atomic_write_json(Path(live_runtime_dir) / LEDGER_FILENAME, {
        "updated_at": (now or datetime.now()).isoformat(timespec="seconds"),
        "codes": normalize_ledger(ledger),
    })


def update_ledger(ledger: dict[str, dict], account_qty: dict[str, int],
                  bot_buy_codes=frozenset(), bot_sell_codes=frozenset()) -> dict[str, dict]:
    """New ledger after one account sync (pure). account_qty: {code: quantity now in the account} (only held symbols).
    bot_buy_codes / bot_sell_codes: symbols for which the bot had a buy / sell order working since the previous sync."""
    old = normalize_ledger(ledger)
    buys = {_norm(c) for c in bot_buy_codes}
    sells = {_norm(c) for c in bot_sell_codes}
    out: dict[str, dict] = {}
    for code, qty in account_qty.items():
        code = _norm(code)
        try:
            new_qty = int(qty)
        except (TypeError, ValueError):
            continue
        if new_qty <= 0:
            continue
        prev = old.get(code) or {"acct_qty": 0, "bot_qty": 0}
        bot = int(prev["bot_qty"])
        delta = new_qty - int(prev["acct_qty"])
        if delta > 0:
            if code in buys:
                bot += delta
        elif delta < 0:
            drop = -delta
            if code in sells:
                bot = max(0, bot - drop)                 # the bot sold: its part shrinks first
            else:
                manual = int(prev["acct_qty"]) - bot      # sold by hand: the manual part shrinks first
                bot -= max(0, drop - manual)
        out[code] = {"acct_qty": new_qty, "bot_qty": max(0, min(bot, new_qty))}
    return out


def seed_ledger(positions: dict[str, dict], live_state: dict, live_runtime_dir: Path, date_str: str) -> dict[str, dict]:
    """First ledger (no file yet). A holding is the bot's - with its whole account quantity - when the live state of today
    or of one of the last SEED_LOOKBACK_FILES earlier days shows a bot buy for it; everything else is manual."""
    bot_codes: set[str] = set()

    def collect(state: dict) -> None:
        for c in (state.get("traded_today") or []):
            bot_codes.add(_norm(c))
        for c, meta in (state.get("positions_meta") or {}).items():
            if isinstance(meta, dict) and str(meta.get("buy_session") or "synced").lower() != "synced":
                bot_codes.add(_norm(c))
        for c, st in (state.get("reentry") or {}).items():
            if isinstance(st, dict) and st.get("bot"):
                bot_codes.add(_norm(c))

    collect(live_state or {})
    try:
        days = sorted(p for p in Path(live_runtime_dir).glob("[0-9]" * 8 + ".json") if p.stem < str(date_str))
    except OSError:
        days = []
    for path in days[-SEED_LOOKBACK_FILES:]:
        raw = _read_json(path)
        if raw:
            collect(raw)
    out: dict[str, dict] = {}
    for code, pos in (positions or {}).items():
        try:
            qty = int(pos.get("quantity", 0) or 0)
        except (TypeError, ValueError, AttributeError):
            continue
        if qty > 0:
            code = _norm(code)
            out[code] = {"acct_qty": qty, "bot_qty": qty if code in bot_codes else 0}
    return out


def bot_quantity(ledger: dict[str, dict] | None, code) -> int:
    row = (ledger or {}).get(_norm(code))
    return int(row.get("bot_qty", 0) or 0) if isinstance(row, dict) else 0


def _iso(value) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def build_snapshot(positions: dict[str, dict], ledger: dict[str, dict], names: dict[str, str], date_str: str,
                   auto_sell_scope: str, now: datetime | None = None) -> dict:
    """Display snapshot of the whole account for r016 `holdings` (schema 1)."""
    rows = []
    for code, pos in sorted((positions or {}).items()):
        try:
            qty = int(pos.get("quantity", 0) or 0)
        except (TypeError, ValueError, AttributeError):
            continue
        if qty <= 0:
            continue
        code = _norm(code)
        bot = max(0, min(bot_quantity(ledger, code), qty))
        buy_time = _iso(pos.get("buy_time"))
        rows.append({
            "code": code,
            "name": str(names.get(code) or ""),
            "qty": qty,
            "bot_qty": bot,
            "manual_qty": qty - bot,
            "avg_price": float(pos.get("buy_price") or 0.0),
            "current_price": float(pos.get("current_price") or 0.0),
            "buy_time": buy_time,
            "bought_today": bool(buy_time and buy_time[:10].replace("-", "") == str(date_str)),
        })
    return {
        "schema": 1,
        "updated_at": (now or datetime.now()).isoformat(timespec="seconds"),
        "date": str(date_str),
        "auto_sell_scope": str(auto_sell_scope),
        "holdings": rows,
    }


def save_snapshot(live_runtime_dir: Path, snapshot: dict) -> None:
    _atomic_write_json(Path(live_runtime_dir) / SNAPSHOT_FILENAME, snapshot)


def load_snapshot(live_runtime_dir: Path) -> dict | None:
    raw = _read_json(Path(live_runtime_dir) / SNAPSHOT_FILENAME)
    if raw is None or not isinstance(raw.get("holdings"), list):
        return None
    return raw
