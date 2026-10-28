# -*- coding: utf-8 -*-
"""g011_tick_collector.py - record KIS real-time executions (H0STCNT0, KRX) per print for 60/120-tick research.

Why a separate process
- r003 is REST-polling only; historical "10s" files are interpolated 1m bars, so no real tick data exists.
  Strategy B/C (120/60-tick filter) cannot be backtested until ticks are collected going forward.
- This process places NO orders and does not touch the REST token file (only /oauth2/Approval for the
  websocket approval key), so it can run next to r003.

Integrity
- The KIS sample parser (kis_auth.KISWebSocket) reads a multi-record frame ("0|H0STCNT0|003|a^b^...")
  with pandas as ONE row, so bundled prints can be lost or mis-aligned. Here the frame is split by the
  record count field. Each record is written as-is with the receive timestamp; dedup / reorder / gap handling
  happens at read time in r009_tick_bars.TickBarBuilder (ACML_VOL + cumulative execution counts).
- Reconnect with exponential backoff; every reconnect is logged so downstream can mark gaps.
- KIS limit: <= 40 subscriptions per websocket session -> codes are capped at 40.

Output: <runtime>/data/ticks/YYYYMMDD/<code>.csv (columns = r009_tick_bars.TICK_FILE_COLUMNS)
        <runtime>/data/ticks/YYYYMMDD/_collector.log

Run:
  python g011_tick_collector.py                       # codes from live_trading/r004_trade_watchlist_today.txt
  python g011_tick_collector.py --codes 089890 253590 --until 15:31
  python g011_tick_collector.py --dry-parse-test      # offline parser self-test, no network

Update log:
- [2026-10-23] type=feat owner=claude
    summary: 신규 - KIS 웹소켓 실시간체결(H0STCNT0) 건별 수집기(다건 프레임 분리, 재접속, 40종목 상한).
    impact: collector (주문 없음, r003과 독립 실행)
    compatibility: backward-compatible (신규 파일)
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIVE_DIR = HERE.parent / "live_trading"
sys.path.insert(0, str(LIVE_DIR))

from r001_define_config import RUNTIME_DATA_DIR, DEFINE_TODAY_CODE_PATH  # noqa: E402
from r009_tick_bars import TICK_FILE_COLUMNS  # noqa: E402

# H0STCNT0 record layout (open-trading-api domestic_stock_functions_ws.ccnl_krx)
H0STCNT0_COLUMNS = [
    "MKSC_SHRN_ISCD", "STCK_CNTG_HOUR", "STCK_PRPR", "PRDY_VRSS_SIGN",
    "PRDY_VRSS", "PRDY_CTRT", "WGHN_AVRG_STCK_PRC", "STCK_OPRC",
    "STCK_HGPR", "STCK_LWPR", "ASKP1", "BIDP1", "CNTG_VOL", "ACML_VOL",
    "ACML_TR_PBMN", "SELN_CNTG_CSNU", "SHNU_CNTG_CSNU", "NTBY_CNTG_CSNU",
    "CTTR", "SELN_CNTG_SMTN", "SHNU_CNTG_SMTN", "CCLD_DVSN", "SHNU_RATE",
    "PRDY_VOL_VRSS_ACML_VOL_RATE", "OPRC_HOUR", "OPRC_VRSS_PRPR_SIGN",
    "OPRC_VRSS_PRPR", "HGPR_HOUR", "HGPR_VRSS_PRPR_SIGN", "HGPR_VRSS_PRPR",
    "LWPR_HOUR", "LWPR_VRSS_PRPR_SIGN", "LWPR_VRSS_PRPR", "BSOP_DATE",
    "NEW_MKOP_CLS_CODE", "TRHT_YN", "ASKP_RSQN1", "BIDP_RSQN1",
    "TOTAL_ASKP_RSQN", "TOTAL_BIDP_RSQN", "VOL_TNRT",
    "PRDY_SMNS_HOUR_ACML_VOL", "PRDY_SMNS_HOUR_ACML_VOL_RATE",
    "HOUR_CLS_CODE", "MRKT_TRTM_CLS_CODE", "VI_STND_PRC",
]
NCOL = len(H0STCNT0_COLUMNS)
_IDX = {c: i for i, c in enumerate(H0STCNT0_COLUMNS)}
MAX_SUBSCRIPTIONS = 40
TR_ID = "H0STCNT0"


def parse_frame(raw: str) -> list[dict] | None:
    """Split one data frame into records. Returns None for non-data (JSON control) frames.
    Format: <encrypt flag>|<tr_id>|<record count>|<f1^f2^...> (count x NCOL fields)."""
    if not raw or raw[0] not in "01":
        return None
    parts = raw.split("|", 3)
    if len(parts) < 4 or parts[1] != TR_ID:
        return []
    if parts[0] == "1":
        raise ValueError("encrypted H0STCNT0 frame is not expected")
    try:
        count = int(parts[2])
    except ValueError:
        return []
    fields = parts[3].split("^")
    # Some feeds append extra columns (e.g. MARKET_CLS_CODE in newer specs): infer width from the total.
    width = len(fields) // count if count > 0 and len(fields) % count == 0 else NCOL
    if width < NCOL:
        return []
    out = []
    for i in range(count):
        rec = fields[i * width:(i + 1) * width]
        out.append({
            "code": rec[_IDX["MKSC_SHRN_ISCD"]],
            "bsop_date": rec[_IDX["BSOP_DATE"]],
            "trade_time": rec[_IDX["STCK_CNTG_HOUR"]],
            "price": rec[_IDX["STCK_PRPR"]],
            "cntg_vol": rec[_IDX["CNTG_VOL"]],
            "acml_vol": rec[_IDX["ACML_VOL"]],
            "ccld_dvsn": rec[_IDX["CCLD_DVSN"]],
            "seln_cntg_csnu": rec[_IDX["SELN_CNTG_CSNU"]],
            "shnu_cntg_csnu": rec[_IDX["SHNU_CNTG_CSNU"]],
            "askp1": rec[_IDX["ASKP1"]],
            "bidp1": rec[_IDX["BIDP1"]],
            "hour_cls_code": rec[_IDX["HOUR_CLS_CODE"]],
        })
    return out


class TickWriter:
    """Append-only per-code CSV writer, flushed periodically. One file handle per code."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._files: dict[str, tuple] = {}
        self.rows = 0

    def write(self, recv_ts: str, rec: dict) -> None:
        code = str(rec["code"]).zfill(6)
        fh = self._files.get(code)
        if fh is None:
            path = self.out_dir / f"{code}.csv"
            new = not path.exists() or path.stat().st_size == 0
            f = open(path, "a", newline="", encoding="utf-8")
            w = csv.DictWriter(f, fieldnames=TICK_FILE_COLUMNS)
            if new:
                w.writeheader()
            fh = (f, w)
            self._files[code] = fh
        fh[1].writerow({"recv_ts": recv_ts, **rec, "code": code})
        self.rows += 1

    def flush(self) -> None:
        for f, _w in self._files.values():
            f.flush()

    def close(self) -> None:
        for f, _w in self._files.values():
            try:
                f.flush()
                f.close()
            except Exception:
                pass
        self._files.clear()


def load_codes(args) -> list[str]:
    if args.codes:
        codes = [c.zfill(6) for c in args.codes]
    else:
        path = LIVE_DIR / DEFINE_TODAY_CODE_PATH
        codes = []
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = [p.strip() for p in s.split(",")]
            code = parts[1] if len(parts) >= 3 and len(parts[0]) == 8 else parts[0]
            if code.isdigit():
                codes.append(code.zfill(6))
    seen, uniq = set(), []
    for c in codes:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq[:MAX_SUBSCRIPTIONS]


async def run_collector(codes: list[str], until: datetime, out_dir: Path, log) -> None:
    import websockets  # imported lazily so --dry-parse-test works without network deps

    open_api_root = Path(os.environ.get("OPEN_TRADING_API_ROOT", str(Path.home() / "git" / "open-trading-api")))
    sys.path.insert(0, str(open_api_root / "examples_user"))
    sys.path.insert(0, str(open_api_root / "examples_user" / "domestic_stock"))
    import kis_auth as ka  # noqa: E402
    from domestic_stock_functions_ws import ccnl_krx  # noqa: E402

    writer = TickWriter(out_dir)
    backoff = 1.0
    stop = asyncio.Event()

    def _stop(*_a):
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            pass

    try:
        while not stop.is_set() and datetime.now() < until:
            try:
                ka.auth_ws()
                url = f"{ka.getTREnv().my_url_ws}/tryitout/{TR_ID}"
                async with websockets.connect(url, ping_interval=None, max_queue=4096) as ws:
                    for code in codes:
                        msg, _cols = ccnl_krx("1", code)
                        await ws.send(json.dumps(msg))
                        await asyncio.sleep(0.05)
                    log(f"CONNECTED url={url} requested={len(codes)}")
                    sub_ok: set = set()
                    sub_fail: set = set()
                    sub_reported = False
                    backoff = 1.0
                    last_flush = time.monotonic()
                    while not stop.is_set() and datetime.now() < until:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        except asyncio.TimeoutError:
                            writer.flush()
                            continue
                        recv_ts = datetime.now().isoformat(timespec="milliseconds")
                        recs = parse_frame(raw)
                        if recs is None:
                            try:
                                ctl = json.loads(raw)
                            except ValueError:
                                log(f"UNKNOWN FRAME {raw[:120]}")
                                continue
                            tr_id = ctl.get("header", {}).get("tr_id")
                            if tr_id == "PINGPONG":
                                await ws.pong(raw)
                            else:
                                body = ctl.get("body") or {}
                                key = ctl.get("header", {}).get("tr_key")
                                if body.get("rt_cd") not in (None, "0"):
                                    sub_fail.add(key)
                                    log(f"SUBSCRIBE ERROR {key} {body.get('msg1')}")
                                elif body.get("rt_cd") == "0":
                                    sub_ok.add(key)
                                if len(sub_ok) + len(sub_fail) == len(codes) and not sub_reported:
                                    sub_reported = True
                                    log(f"SUBSCRIPTIONS ok={len(sub_ok)} failed={len(sub_fail)} {sorted(sub_fail)}")
                            continue
                        for rec in recs:
                            writer.write(recv_ts, rec)
                        if time.monotonic() - last_flush >= 1.0:
                            writer.flush()
                            last_flush = time.monotonic()
            except Exception as exc:  # network / auth errors: back off and reconnect (gap is visible via ACML_VOL)
                writer.flush()
                log(f"DISCONNECTED {type(exc).__name__}: {exc} | retry in {backoff:.0f}s")
                try:
                    await asyncio.wait_for(stop.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2.0, 60.0)
    finally:
        writer.close()
        log(f"STOPPED rows={writer.rows}")


def _self_test() -> None:
    def rec(code, hhmmss, px, q, acml, side, s, b):
        f = [""] * NCOL
        f[_IDX["MKSC_SHRN_ISCD"]] = code
        f[_IDX["STCK_CNTG_HOUR"]] = hhmmss
        f[_IDX["STCK_PRPR"]] = str(px)
        f[_IDX["CNTG_VOL"]] = str(q)
        f[_IDX["ACML_VOL"]] = str(acml)
        f[_IDX["CCLD_DVSN"]] = str(side)
        f[_IDX["SELN_CNTG_CSNU"]] = str(s)
        f[_IDX["SHNU_CNTG_CSNU"]] = str(b)
        f[_IDX["BSOP_DATE"]] = "20261023"
        return f
    fields = rec("005930", "090001", 100, 5, 5, 1, 0, 1) + rec("005930", "090001", 101, 3, 8, 1, 0, 2)
    frame = f"0|{TR_ID}|002|" + "^".join(fields)
    out = parse_frame(frame)
    assert out is not None and len(out) == 2, out
    assert out[1]["price"] == "101" and out[1]["acml_vol"] == "8"
    assert parse_frame('{"header":{"tr_id":"PINGPONG"}}') is None
    print("parse_frame self-test OK (2 bundled records split correctly)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--codes", nargs="*", default=None)
    ap.add_argument("--until", default="15:31", help="HH:MM local time to stop")
    ap.add_argument("--dry-parse-test", action="store_true")
    args = ap.parse_args()
    if args.dry_parse_test:
        _self_test()
        return
    codes = load_codes(args)
    if not codes:
        print("no codes to subscribe")
        return
    now = datetime.now()
    hh, mm = (int(x) for x in args.until.split(":"))
    until = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    out_dir = RUNTIME_DATA_DIR / "ticks" / now.strftime("%Y%m%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "_collector.log"

    def log(msg: str) -> None:
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"START codes={len(codes)} until={until:%H:%M} out={out_dir}")
    asyncio.run(run_collector(codes, until, out_dir, log))


if __name__ == "__main__":
    main()
