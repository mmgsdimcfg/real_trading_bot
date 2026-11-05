# -*- coding: utf-8 -*-
"""g025_daily_history.py - 종목별 일봉 이력 파일 runtime/data/daily_history/<code>.csv (AutoBotH 일/주/월봉용, 읽기 전용 출력).

Why: g001 은 날짜 폴더마다 전 종목 _daily.csv(200행)를 다시 쓴다(폴더당 ~1.1 GB). 차트는 종목 하나의 긴 이력만 필요하다.
What: 날짜 폴더들의 {code}_{name}_daily.csv 를 합쳐 종목당 한 파일로 유지한다. 같은 날짜 행은 더 늦은 폴더의 값이 이긴다
      (g001 Update log 2026-10-10: 당일 폴더의 당일 행은 16시대 미확정 관측, 다음날 폴더 행이 공식 종가).
      열: date,open,high,low,close,volume,amount,source_date   source_date = 그 행을 제공한 폴더 날짜.
      source_date == date 인 행은 아직 미확정(provisional) 이다.
Run:
  python g025_daily_history.py --rebuild            # 모든 날짜 폴더에서 재구성(종목당 가장 오래된/최신 폴더 + 200행 간격 폴더만 읽음)
  python g025_daily_history.py --append 20261105    # 그 날짜 폴더의 행을 upsert (장 마감 g001 뒤 cron)
  python g025_daily_history.py --append             # 가장 최근 폴더
권장 cron: 0 21 * * 1-5  python g025_daily_history.py --append >> runtime/log/g025_cron.log 2>&1
주의: KIS API 를 호출하지 않는다(폴더에 있는 것만). 3년 이력은 폴더 범위(2025-09~)에 한정 - 더 긴 이력은 g001 --lookback 확대 후 --rebuild.

Update log:
- [2026-11-06] type=feat owner=claude
    summary: AutoBotH 차트 요청(P3-2) 신규.
    impact: tooling
    compatibility: backward-compatible (신규 파일)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "live_trading"))
with contextlib.redirect_stdout(io.StringIO()):
    import r001_define_config as r001  # noqa: E402

COLS = ["date", "open", "high", "low", "close", "volume", "amount"]
DAILY_RE = re.compile(r"^(\d{6})_.*_daily\.csv$")
ROWS_PER_FILE = 200


def data_root() -> Path:
    return Path(os.environ.get("AUTO_TRADING_RUNTIME") or r001.RUNTIME_ROOT) / "data"


def history_dir() -> Path:
    return data_root() / "daily_history"


def date_folders() -> list[str]:
    return sorted(p.name for p in data_root().iterdir() if p.is_dir() and re.fullmatch(r"\d{8}", p.name))


def read_daily_file(path: Path, source_date: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype={"date": str}, usecols=lambda c: c in COLS)
    except Exception:
        return None
    if "date" not in df.columns or "close" not in df.columns:
        return None
    for c in COLS[1:]:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[COLS].copy()
    df["date"] = df["date"].astype(str).str.strip()
    df = df[df["date"].str.fullmatch(r"\d{8}")]
    df["source_date"] = source_date
    return df


def merge_rows(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Later source_date wins for the same date; a provisional same-day row never overrides a later folder's row."""
    frames = [f for f in (existing, new) if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=COLS + ["source_date"])
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["date", "source_date"], kind="stable").drop_duplicates(subset=["date"], keep="last")
    return df.sort_values("date").reset_index(drop=True)


def read_history(code: str) -> pd.DataFrame | None:
    p = history_dir() / f"{code}.csv"
    if not p.is_file():
        return None
    try:
        return pd.read_csv(p, dtype={"date": str, "source_date": str}, encoding="utf-8")
    except Exception:
        return None


def write_history(code: str, df: pd.DataFrame) -> None:
    history_dir().mkdir(parents=True, exist_ok=True)
    p = history_dir() / f"{code}.csv"
    tmp = p.with_suffix(f".csv.{os.getpid()}.tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, p)


def folder_files(folder: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    d = data_root() / folder
    if not d.is_dir():
        return out
    for p in d.iterdir():
        m = DAILY_RE.match(p.name)
        if m:
            out[m.group(1)] = p
    return out


def append_folder(folder: str, log=print) -> int:
    files = folder_files(folder)
    n = 0
    for code, path in sorted(files.items()):
        new = read_daily_file(path, folder)
        if new is None or new.empty:
            continue
        write_history(code, merge_rows(read_history(code), new))
        n += 1
    log(f"{datetime.now():%F %T} g025 append {folder}: {n} codes")
    return n


def rebuild(log=print, step: int = ROWS_PER_FILE // 2) -> int:
    """Read the oldest folder, then every `step`-th folder, then the newest two (each file holds 200 rows, so folders
    ~100 trading days apart already overlap). Provisional same-day rows are replaced by the next folder's final rows."""
    folders = date_folders()
    if not folders:
        log("g025 rebuild: no date folders")
        return 0
    picked = sorted(set(folders[::step]) | set(folders[-2:]) | {folders[0]})
    log(f"{datetime.now():%F %T} g025 rebuild from {len(picked)}/{len(folders)} folders: {picked}")
    merged: dict[str, pd.DataFrame] = {}
    for folder in picked:
        t0 = time.time()
        for code, path in folder_files(folder).items():
            new = read_daily_file(path, folder)
            if new is None or new.empty:
                continue
            merged[code] = merge_rows(merged.get(code), new)
        log(f"  {folder}: {len(merged)} codes so far ({time.time() - t0:.0f}s)")
    for code, df in merged.items():
        write_history(code, df)
    log(f"{datetime.now():%F %T} g025 rebuild done: {len(merged)} codes -> {history_dir()}")
    return len(merged)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rebuild", action="store_true")
    g.add_argument("--append", nargs="?", const="latest", default=None, metavar="YYYYMMDD")
    args = ap.parse_args(argv)
    if args.rebuild:
        return 0 if rebuild() else 1
    folder = args.append
    if folder == "latest":
        folders = date_folders()
        if not folders:
            print("g025: no date folders")
            return 1
        folder = folders[-1]
    done_marker = data_root() / folder / "_g001_done.json"
    if not done_marker.is_file():
        print(f"g025: {folder} has no _g001_done.json (g001 not finished) - appending what exists anyway")
    return 0 if append_folder(folder) else 1


if __name__ == "__main__":
    sys.exit(main())
