# -*- coding: utf-8 -*-
"""G005 watchlist bridge: g002 scanner picks -> r004 live watchlist.

Feature-gated by r001_define_config:
- FEATURE_R004_WATCHLIST
- FEATURE_SCAN_EXPORT_TO_R004
- FEATURE_WATCHLIST_RESOLVE_SCAN_PICKS

Update log:
- [2026-11-28] type=fix owner=claude
    summary: (1) resolve_watchlist_path(target_date): the picks of the newest scan BEFORE the date (was the date's own scan,
      made after that day's close = look-ahead for `r003 --date D` replays; the live service passes no date and reads r004).
      (2) for_trading_day header of a direct export = the next trading day (was calendar + 1: Friday -> Saturday).
    impact: sim / replay, the header of a direct g002 export (the r021 pipeline already used the trading calendar)
    compatibility: backward-compatible
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "live_trading"))  # r001_define_config

try:
    from r001_define_config import (
        FEATURE_R004_WATCHLIST,
        FEATURE_SCAN_EXPORT_TO_R004,
        FEATURE_WATCHLIST_RESOLVE_SCAN_PICKS,
        R004_WATCHLIST_FILENAME,
        SCAN_PICKS_LEGACY_FILENAME,
        SCAN_PICKS_PREFIX_TEMPLATE,
        DATA_DIR_NAME,
        RUNTIME_DATA_DIR,
    )
except Exception:
    FEATURE_R004_WATCHLIST = True
    FEATURE_SCAN_EXPORT_TO_R004 = True
    FEATURE_WATCHLIST_RESOLVE_SCAN_PICKS = True
    R004_WATCHLIST_FILENAME = "r004_trade_watchlist_today.txt"
    SCAN_PICKS_LEGACY_FILENAME = "picks.txt"
    SCAN_PICKS_PREFIX_TEMPLATE = "_{date}_picks.txt"
    DATA_DIR_NAME = "data"
    RUNTIME_DATA_DIR = Path(os.environ.get("AUTO_TRADING_RUNTIME") or Path(__file__).resolve().parents[3] / "runtime") / DATA_DIR_NAME

R004_HEADER = """# Daily live trading watchlist.
#
# Purpose:
# - List of symbols used by r003_trade_live_execute.py and optional simulation watchlist mode.
#
# Format:
# - code,name
# - Example: 005930,삼성전자
#
# [FEATURE_R004_WATCHLIST] auto-generated from g002 scanner export.
"""


def scan_picks_path(data_dir: Path, scan_date: str) -> Path:
    return data_dir / scan_date / SCAN_PICKS_PREFIX_TEMPLATE.format(date=scan_date)


def legacy_picks_path(data_dir: Path, scan_date: str) -> Path:
    return data_dir / scan_date / SCAN_PICKS_LEGACY_FILENAME


def next_trading_day(scan_date: str) -> str | None:
    """[2026-11-28] the trading day after scan_date (YYYYMMDD) by the bot's calendar (r021.is_trading_day = r014), None when
    the date is bad. Was calendar + 1 (a Friday scan was labelled Saturday)."""
    try:
        d = datetime.strptime(scan_date, "%Y%m%d")
    except (TypeError, ValueError):
        return None
    try:
        from r021_jobs import is_trading_day
    except Exception:
        def is_trading_day(ymd: str) -> bool:  # weekends only
            return datetime.strptime(ymd, "%Y%m%d").weekday() < 5
    for _ in range(14):
        d += timedelta(days=1)
        if is_trading_day(d.strftime("%Y%m%d")):
            return d.strftime("%Y%m%d")
    return None


def prior_scan_picks_path(data_root: Path, target_date: str, legacy: bool = False, max_back: int = 10) -> Path | None:
    """[2026-11-28] the picks file a trading day really had: the one of the newest scan date BEFORE target_date (the scan of
    target_date itself runs after that day's close - reading it for that day is look-ahead). Looks at the max_back newest
    date folders before target_date; None when none of them has a picks file."""
    try:
        days = sorted((p.name for p in Path(data_root).iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 8 and p.name < target_date), reverse=True)
    except OSError:
        return None
    for d in days[:max_back]:
        p = legacy_picks_path(data_root, d) if legacy else scan_picks_path(data_root, d)
        if p.exists():
            return p
    return None


def resolve_watchlist_path(
    auto_trading_dir: Path,
    target_date: str | None = None,
    watchlist_source: str = "auto",
    data_dir: Path | None = None,
) -> Path:
    """Resolve watchlist file for r006 / simulation."""
    data_root = data_dir or RUNTIME_DATA_DIR  # [2026-10-11] <git root>/runtime/data
    r004_path = auto_trading_dir / R004_WATCHLIST_FILENAME
    source = (watchlist_source or "auto").strip().lower()

    if source == "auto":
        if target_date:
            source = "scan-picks" if FEATURE_WATCHLIST_RESOLVE_SCAN_PICKS else "picks"
        else:
            source = "r004" if FEATURE_R004_WATCHLIST else "r004"

    if source == "r004" or not target_date:
        return r004_path

    # [2026-11-28] the list of trading day D is the scan made before D (was: D's own scan = that day's close known in advance)
    if source == "scan-picks":
        scan_path = prior_scan_picks_path(data_root, target_date)
        if scan_path is not None:
            return scan_path
    legacy = prior_scan_picks_path(data_root, target_date, legacy=True)
    if legacy is not None:
        return legacy
    return r004_path


def render_r004_content(
    picks_lines: list[str],
    scan_date: str | None = None,
    config_name: str | None = None,
    for_next_trading_day: bool = False,
) -> str:
    meta = []
    if scan_date:
        meta.append(f"# scan_date={scan_date}")
    if for_next_trading_day and scan_date:
        nxt = next_trading_day(scan_date)
        if nxt:
            meta.append(f"# for_trading_day={nxt} (next trading day)")
    if config_name:
        meta.append(f"# scanner_config={config_name}")
    meta.append("# [FEATURE_SCAN_EXPORT_TO_R004]")

    body = [ln.strip() for ln in picks_lines if ln.strip() and not ln.strip().startswith("#")]
    return R004_HEADER + "\n".join(meta) + "\n\n" + "\n".join(body) + "\n"


def export_picks_to_r004(
    picks_lines: list[str],
    auto_trading_dir: Path,
    scan_date: str | None = None,
    config_name: str | None = None,
    for_next_trading_day: bool = False,
    out_path: Path | None = None,
) -> Path | None:
    if not FEATURE_SCAN_EXPORT_TO_R004:
        return None
    if not picks_lines:
        return None
    dest = out_path or (auto_trading_dir / R004_WATCHLIST_FILENAME)
    dest.write_text(
        render_r004_content(
            picks_lines,
            scan_date=scan_date,
            config_name=config_name,
            for_next_trading_day=for_next_trading_day,
        ),
        encoding="utf-8",
    )
    return dest


def write_legacy_picks_alias(
    picks_lines: list[str],
    data_dir: Path,
    scan_date: str,
) -> Path | None:
    """Also write data/YYYYMMDD/picks.txt for backward compatibility with r006 --date."""
    if not picks_lines:
        return None
    out_dir = data_dir / scan_date
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = legacy_picks_path(data_dir, scan_date)
    dest.write_text("\n".join(picks_lines) + "\n", encoding="utf-8")
    return dest
