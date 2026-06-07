# -*- coding: utf-8 -*-
"""R010 watchlist bridge: r002 scanner picks -> r008 live watchlist.

Feature-gated by r003_define_config:
- FEATURE_R008_WATCHLIST
- FEATURE_SCAN_EXPORT_TO_R008
- FEATURE_WATCHLIST_RESOLVE_SCAN_PICKS
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

try:
    from r003_define_config import (
        FEATURE_R008_WATCHLIST,
        FEATURE_SCAN_EXPORT_TO_R008,
        FEATURE_WATCHLIST_RESOLVE_SCAN_PICKS,
        R008_WATCHLIST_FILENAME,
        SCAN_PICKS_LEGACY_FILENAME,
        SCAN_PICKS_PREFIX_TEMPLATE,
        DATA_DIR_NAME,
    )
except Exception:
    FEATURE_R008_WATCHLIST = True
    FEATURE_SCAN_EXPORT_TO_R008 = True
    FEATURE_WATCHLIST_RESOLVE_SCAN_PICKS = True
    R008_WATCHLIST_FILENAME = "r008_trade_watchlist_today.txt"
    SCAN_PICKS_LEGACY_FILENAME = "picks.txt"
    SCAN_PICKS_PREFIX_TEMPLATE = "_{date}_picks.txt"
    DATA_DIR_NAME = "data"

R008_HEADER = """# R76 daily live trading watchlist.
#
# Purpose:
# - List of symbols used by r006_trade_live_execute.py and optional simulation watchlist mode.
#
# Format:
# - code,name
# - Example: 005930,삼성전자
#
# [FEATURE_R008_WATCHLIST] auto-generated from r002 scanner export.
"""


def scan_picks_path(data_dir: Path, scan_date: str) -> Path:
    return data_dir / scan_date / SCAN_PICKS_PREFIX_TEMPLATE.format(date=scan_date)


def legacy_picks_path(data_dir: Path, scan_date: str) -> Path:
    return data_dir / scan_date / SCAN_PICKS_LEGACY_FILENAME


def resolve_watchlist_path(
    auto_trading_dir: Path,
    target_date: str | None = None,
    watchlist_source: str = "auto",
    data_dir: Path | None = None,
) -> Path:
    """Resolve watchlist file for r006 / simulation."""
    data_root = data_dir or (auto_trading_dir / DATA_DIR_NAME)
    r008_path = auto_trading_dir / R008_WATCHLIST_FILENAME
    source = (watchlist_source or "auto").strip().lower()

    if source == "auto":
        if target_date:
            source = "scan-picks" if FEATURE_WATCHLIST_RESOLVE_SCAN_PICKS else "picks"
        else:
            source = "r008" if FEATURE_R008_WATCHLIST else "r008"

    if source == "r008" or not target_date:
        return r008_path

    date_dir = data_root / target_date
    if source == "scan-picks":
        scan_path = scan_picks_path(data_root, target_date)
        if scan_path.exists():
            return scan_path
    legacy = legacy_picks_path(data_root, target_date)
    if legacy.exists():
        return legacy
    return r008_path


def render_r008_content(
    picks_lines: list[str],
    scan_date: str | None = None,
    config_name: str | None = None,
    for_next_trading_day: bool = False,
) -> str:
    meta = []
    if scan_date:
        meta.append(f"# scan_date={scan_date}")
    if for_next_trading_day and scan_date:
        try:
            d = datetime.strptime(scan_date, "%Y%m%d")
            meta.append(f"# for_trading_day={(d + timedelta(days=1)).strftime('%Y%m%d')} (calendar+1, verify market day)")
        except ValueError:
            pass
    if config_name:
        meta.append(f"# scanner_config={config_name}")
    meta.append("# [FEATURE_SCAN_EXPORT_TO_R008]")

    body = [ln.strip() for ln in picks_lines if ln.strip() and not ln.strip().startswith("#")]
    return R008_HEADER + "\n".join(meta) + "\n\n" + "\n".join(body) + "\n"


def export_picks_to_r008(
    picks_lines: list[str],
    auto_trading_dir: Path,
    scan_date: str | None = None,
    config_name: str | None = None,
    for_next_trading_day: bool = False,
    out_path: Path | None = None,
) -> Path | None:
    if not FEATURE_SCAN_EXPORT_TO_R008:
        return None
    if not picks_lines:
        return None
    dest = out_path or (auto_trading_dir / R008_WATCHLIST_FILENAME)
    dest.write_text(
        render_r008_content(
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
