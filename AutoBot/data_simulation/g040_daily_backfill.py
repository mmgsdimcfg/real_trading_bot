# -*- coding: utf-8 -*-
"""g040_daily_backfill.py - long daily-bar history for research (daily swing / market-regime studies), collected slowly in the
background on the standby host (Raspberry Pi) and copied to the notebook afterwards.

Why: runtime/data/daily_history (g025) starts 2025-09 (the g001 date folders) - one bull and one bear market only. The daily swing
study (runtime/research/08_daily_swing) needs several years incl. sideways markets before the market filter can be trusted.

What: for every code of g004_universe_symbols_master.txt (commented-out codes too: they were listed once) + the index ETFs
(r001 MARKET_REGIME_ETF), page backwards with KIS inquire_daily_itemchartprice (100 rows per call, adjusted prices
fid_org_adj_prc="0" - the same as g001) from the newest finished trading day to --since.
  Output: <runtime>/data/daily_backfill/<code>.csv  date,open,high,low,close,volume,amount  (oldest first)
          <runtime>/data/daily_backfill/_progress.json  {code: {"oldest": d, "newest": d, "done": bool, "rows": n, "at": ts}}
          <runtime>/log/g040_daily_backfill.log
  Resumable: a code whose file already reaches --since (or the listing date: a short last page) is skipped; a code that stopped
  half-way continues from its oldest row. --refresh-newest also tops up rows after the newest stored date.
  It never writes daily_history (the bot's / app's file) - research reads both folders.

Load: one call at a time, at most --rate calls per second (default 2), only inside --window (default 20:10-07:40 on weekdays, all day
on weekends; KST) so the live bot on the other host keeps the REST rate limit during the regular and NXT sessions (08:00-20:00). Outside the window it sleeps.
About 20 calls per code for 8 years -> ~3,600 codes x 20 = ~72k calls = ~10 hours at 2/s (about one night + a weekend).
Single instance: <runtime>/data/daily_backfill/.g040.lock (a second start exits with rc 3) - a cron line can simply restart it.

Run (Pi, standby host):
  /home/xgraph/venv/bin/python data_simulation/g040_daily_backfill.py --since 20180101
  crontab (restart if it died; it sleeps outside the window by itself):
    */20 * * * *  cd /data/srv/autobot/real_trading_bot/AutoBot && /home/xgraph/venv/bin/python data_simulation/g040_daily_backfill.py --since 20180101 >> /data/srv/autobot/runtime/log/g040_cron.log 2>&1
  status:  python data_simulation/g040_daily_backfill.py --status
Copy to the notebook (PowerShell, OpenSSH):
  scp -r xgraph@<pi-host>:/data/srv/autobot/runtime/data/daily_backfill D:\\make_utils\\Trading\\runtime\\data\\
  (or: powershell -ExecutionPolicy Bypass -File data_simulation/g041_pull_daily_backfill.ps1 -PiHost <pi-host>)

Update log:
- [2026-11-17] type=feat owner=claude
    summary: 신규 - 사용자 요청(일봉 스윙 검증용 과거 일봉). 라즈베리파이에서 장 외 시간에 천천히 받아 노트북으로 복사.
    impact: tooling (KIS REST 조회만, 주문 없음, daily_history 미변경)
    compatibility: backward-compatible (신규 파일)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "live_trading"))
with contextlib.redirect_stdout(io.StringIO()):
    import g001_data_collect_symbols_daily as g001  # noqa: E402  (KIS paths, timeout/throttle session)
import r001_define_config as r001  # noqa: E402
import r019_platform as r019  # noqa: E402

KST = ZoneInfo("Asia/Seoul")
COLS = ["date", "open", "high", "low", "close", "volume", "amount"]
COL_MAP = {"stck_bsop_date": "date", "stck_oprc": "open", "stck_hgpr": "high", "stck_lwpr": "low", "stck_clpr": "close",
           "acml_vol": "volume", "acml_tr_pbmn": "amount"}
PAGE_ROWS = 100
EXIT_ALREADY_RUNNING = 3


def out_dir(runtime: Path | None = None) -> Path:
    return Path(runtime or r001.RUNTIME_ROOT) / "data" / "daily_backfill"


def log_path(runtime: Path | None = None) -> Path:
    return Path(runtime or r001.RUNTIME_ROOT) / "log" / "g040_daily_backfill.log"


def log(msg: str, runtime: Path | None = None) -> None:
    line = f"{datetime.now(KST):%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    p = log_path(runtime)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_codes(symbols_file: Path) -> list[str]:
    """Universe codes incl. commented-out rows ("# 000040,KR모터스,KOSPI  # auto: 관리종목") + index ETFs, unique, file order."""
    codes: list[str] = []
    for raw in symbols_file.read_text(encoding="utf-8-sig").splitlines():
        s = raw.strip().lstrip("#").strip()
        head = s.split(",", 1)[0].strip()
        if len(head) == 6 and head.isalnum() and head != "code":
            codes.append(head)
    for etf in dict(getattr(r001, "MARKET_REGIME_ETF", {})).values():
        codes.insert(0, str(etf))
    return list(dict.fromkeys(codes))


def in_window(now: datetime, window: str) -> bool:
    """window "HH:MM-HH:MM" (may wrap midnight) on weekdays; weekends always allowed."""
    if now.weekday() >= 5:
        return True
    a, b = window.split("-")
    t = now.strftime("%H:%M")
    return (a <= t or t < b) if a > b else (a <= t < b)


class Throttle:
    def __init__(self, rate: float):
        self.gap = 1.0 / max(rate, 0.01)
        self.last = 0.0

    def wait(self) -> None:
        d = self.last + self.gap - time.monotonic()
        if d > 0:
            time.sleep(d)
        self.last = time.monotonic()


def fetch_page(code: str, date_from: str, date_to: str, env_dv: str) -> pd.DataFrame | None:
    from inquire_daily_itemchartprice import inquire_daily_itemchartprice as api
    _, page = api(env_dv=env_dv, fid_cond_mrkt_div_code="J", fid_input_iscd=code, fid_input_date_1=date_from,
                  fid_input_date_2=date_to, fid_period_div_code="D", fid_org_adj_prc="0")
    if page is None or page.empty or "stck_bsop_date" not in page.columns:
        return None
    df = page.rename(columns={k: v for k, v in COL_MAP.items() if k in page.columns})
    df = df[[c for c in COLS if c in df.columns]].copy()
    df["date"] = df["date"].astype(str).str.strip()
    df = df[df["date"].str.fullmatch(r"\d{8}")]
    for c in COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.NA
    return df[(df["close"] > 0)]


def read_existing(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path, dtype={"date": str})
    except Exception:
        return None


def save(path: Path, df: pd.DataFrame) -> None:
    df = df.drop_duplicates("date", keep="last").sort_values("date")[COLS]
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def backfill_code(code: str, since: str, newest: str, env_dv: str, throttle: Throttle, d: Path, prog: dict,
                  refresh_newest: bool, fetch=fetch_page) -> str:
    """Fill one code. Returns "done" | "skip" | "partial" | "empty"."""
    path = d / f"{code}.csv"
    have = read_existing(path)
    st = prog.get(code, {})
    if st.get("done") and not refresh_newest:
        return "skip"
    frames = [have] if have is not None and not have.empty else []
    # top-up after the newest stored row (only with --refresh-newest; new rows come from g025 daily_history anyway)
    if frames and refresh_newest and str(have["date"].max()) < newest:
        throttle.wait()
        top = fetch(code, str(have["date"].max()), newest, env_dv)
        if top is not None:
            frames.append(top)
    page_end = newest if not frames else (datetime.strptime(str(frames[0]["date"].min()), "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
    done = bool(st.get("done"))
    status = "partial"
    while not done and page_end >= since:
        throttle.wait()
        try:
            page = fetch(code, since, page_end, env_dv)
        except Exception as exc:
            log(f"WARN {code} page<= {page_end}: {type(exc).__name__}: {exc}")
            break
        if page is None or page.empty:
            done = True          # nothing older: listing date or before --since
            break
        frames.append(page)
        if len(page) < PAGE_ROWS:
            done = True          # short page = reached the listing date / --since
            break
        oldest = str(page["date"].min())
        if oldest <= since:
            done = True
            break
        page_end = (datetime.strptime(oldest, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
    else:
        done = True
    if not frames:
        prog[code] = {"done": done, "rows": 0, "at": datetime.now(KST).isoformat(timespec="seconds")}
        return "empty"
    df = pd.concat(frames, ignore_index=True)
    df = df[df["date"] >= since]
    if df.empty:
        prog[code] = {"done": done, "rows": 0, "at": datetime.now(KST).isoformat(timespec="seconds")}
        return "empty"
    save(path, df)
    prog[code] = {"oldest": str(df["date"].min()), "newest": str(df["date"].max()), "rows": int(df["date"].nunique()),
                  "done": done, "at": datetime.now(KST).isoformat(timespec="seconds")}
    status = "done" if done else "partial"
    return status


def last_finished_day(now: datetime) -> str:
    """Newest trading day whose close is final: yesterday before 16:00 KST, otherwise today (the API skips holidays itself)."""
    d = now if now.strftime("%H:%M") >= "16:00" else now - timedelta(days=1)
    return d.strftime("%Y%m%d")


def read_progress(d: Path) -> dict:
    with contextlib.suppress(Exception):
        return json.loads((d / "_progress.json").read_text(encoding="utf-8"))
    return {}


def write_progress(d: Path, prog: dict) -> None:
    tmp = d / "_progress.json.tmp"
    tmp.write_text(json.dumps(prog, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")
    os.replace(tmp, d / "_progress.json")


def status(d: Path, codes: list[str], since: str) -> None:
    prog = read_progress(d)
    done = sum(1 for c in codes if prog.get(c, {}).get("done"))
    part = sum(1 for c in codes if c in prog and not prog[c].get("done"))
    rows = sum(int(prog.get(c, {}).get("rows", 0) or 0) for c in codes)
    size = sum(p.stat().st_size for p in d.glob("*.csv")) if d.is_dir() else 0
    oldest = min((v.get("oldest") for v in prog.values() if v.get("oldest")), default="-")
    print(json.dumps({"dir": str(d), "since": since, "codes": len(codes), "done": done, "partial": part,
                      "todo": len(codes) - done - part, "rows": rows, "size_mb": round(size / 1e6, 1), "oldest": oldest},
                     ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser(description="slow KIS daily-bar backfill for research (resumable, off-session window)")
    ap.add_argument("--since", default="20180101", help="oldest date to fetch YYYYMMDD (default 20180101)")
    ap.add_argument("--rate", type=float, default=2.0, help="max KIS calls per second (default 2)")
    ap.add_argument("--window", default="20:10-07:40", help="weekday KST window HH:MM-HH:MM (weekends always); 'always' = no limit")
    ap.add_argument("--code", default="", help="only these codes (comma separated)")
    ap.add_argument("--symbols-file", default=str(HERE / "g004_universe_symbols_master.txt"))
    ap.add_argument("--refresh-newest", action="store_true", help="also top up rows after the newest stored date")
    ap.add_argument("--env", default="real", choices=["real", "demo"])
    ap.add_argument("--status", action="store_true", help="print progress JSON and exit")
    ap.add_argument("--once", action="store_true", help="exit when outside the window instead of sleeping")
    args = ap.parse_args()

    d = out_dir()
    d.mkdir(parents=True, exist_ok=True)
    codes = [c.strip().zfill(6) for c in args.code.split(",") if c.strip()] or load_codes(Path(args.symbols_file))
    if args.status:
        status(d, codes, args.since)
        return 0
    lock = r019.acquire_instance_lock(d / ".g040.lock")   # held for the whole run (flock / msvcrt)
    if lock is None:
        print("g040 already running", flush=True)
        return EXIT_ALREADY_RUNNING
    g001._install_kis_http_session()
    authed = False
    throttle = Throttle(args.rate)
    prog = read_progress(d)
    todo = [c for c in codes if not prog.get(c, {}).get("done") or args.refresh_newest]
    log(f"START codes={len(codes)} todo={len(todo)} since={args.since} rate={args.rate}/s window={args.window} dir={d}")
    counts: dict[str, int] = {}
    for i, code in enumerate(todo, 1):
        while args.window != "always" and not in_window(datetime.now(KST), args.window):
            if args.once:
                log(f"OUTSIDE WINDOW - stop ({i - 1}/{len(todo)})")
                write_progress(d, prog)
                return 0
            time.sleep(60)
        if not authed:
            g001.ka.auth(svr="prod" if args.env == "real" else "vps")
            authed = True
        try:
            st = backfill_code(code, args.since, last_finished_day(datetime.now(KST)), args.env, throttle, d, prog, args.refresh_newest)
        except Exception as exc:  # one bad code must not stop the run
            st = "error"
            log(f"ERROR {code}: {type(exc).__name__}: {exc}")
        counts[st] = counts.get(st, 0) + 1
        if i % 20 == 0 or st in ("partial", "error"):
            write_progress(d, prog)
        if i % 100 == 0:
            log(f"PROGRESS {i}/{len(todo)} {counts}")
    write_progress(d, prog)
    log(f"FINISHED {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
