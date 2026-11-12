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
  A connection that stays open but delivers nothing (no subscribe ACK within SUB_ACK_TIMEOUT_SECONDS, or no
  execution at all for IDLE_RECONNECT_SECONDS during continuous trading) is treated as dead and reconnected.
- KIS limit: <= 40 subscriptions per websocket session -> codes are capped at 40.
- Single instance: the process holds an exclusive flock on <runtime>/data/ticks/.g011.lock for its whole life,
  so r003's supervisor and the g022 cron can never run two collectors (the loser exits with rc 3).
- A row cut by a crash is closed with a newline before appending again, so it cannot swallow the next print
  (r009 then drops the short row).

Output: <runtime>/data/ticks/YYYYMMDD/<code>.csv (columns = r009_tick_bars.TICK_FILE_COLUMNS)
        <runtime>/data/ticks/YYYYMMDD/_collector.log

Run:
  python g011_tick_collector.py                       # codes from live_trading/r004_trade_watchlist_today.txt
  python g011_tick_collector.py --codes 089890 253590 --until 15:31
  python g011_tick_collector.py --dry-parse-test      # offline parser self-test, no network
  python g011_tick_collector.py --fill-from-scan      # r004 codes first, then top-ranked g002 candidates up to 40
  (daily schedule without r003: g022_tick_collect_daily.sh)
  --until defaults to the r014 calendar close + 1 min (15:31 normally, 16:31 on CSAT day); exits on a holiday.

Update log:
- [2026-11-12] type=feat owner=claude
    summary: NXT 체결(H0NXCNT0) 수집 추가 (사용자 요청: NXT 종목은 08:00~20:00 수집). 같은 웹소켓 세션에서 시간대별로 구독을
      바꾼다: KRX(H0STCNT0)는 기존대로 시작~--until, NXT 는 NXT 창(r001 MORNING_NXT_START~AFTERNOON_NXT_END) 동안 감시목록
      종목을 구독하되 세션 한도 40 에서 KRX 가 쓰고 남은 자리만큼만(감시목록 순서). 구독 오류가 난 종목은 NXT 대상에서 뺀다.
      NXT 틱은 ticks/<날짜>/nxt/<종목>.csv 에 따로 쓴다(venue=NXT, session=pre|regular|post) - 기존 <종목>.csv(KRX)는 그대로라
      r013 틱 트리거/틱봉은 바뀌지 않는다. 끄기: r001 TICK_COLLECT_NXT=False 또는 --no-nxt.
    impact: collector (프로세스가 NXT 종료 시각까지 살아 있다), r016 collector 창
    compatibility: backward-compatible (KRX 파일/로그 형식 불변, nxt 하위 폴더 추가)
- [2026-11-10] type=feat owner=claude
    summary: 윈도우 실행 - 단일 실행 잠금을 r019(flock/msvcrt)로, --runtime DIR(r001 import 전에 적용; WMI 로 시작된 프로세스는
      환경변수를 물려받지 않음).
    impact: collector
    compatibility: backward-compatible
- [2026-11-06] type=feat owner=claude
    summary: AutoBotH 차트 요청(P1-1) - 틱 CSV 끝에 venue(KRX), session(pre|regular|close_auction|post) 2열 추가
      (r009 TICK_FILE_COLUMNS v2). session 은 체결시각 vs r014 정규장 (open, close): open 전 pre, [open, close) regular,
      [close, close+10분) close_auction(15:30 동시호가 체결), 그 뒤 post. TickWriter 는 이미 있는 파일은 그 파일 헤더 폭으로
      계속 쓴다(장중 재시작 시 한 파일에 13열/15열 섞이지 않음 - r009/r013 이 폭 불일치 행을 버리므로).
    impact: collector (g011 재시작 후 새 파일부터 15열)
    compatibility: backward-compatible (판독기는 헤더 기반)
- [2026-11-02] type=fix owner=claude
    summary: Codex cron 검토 반영 - (1) 단일 실행 flock(.g011.lock, 실패 시 rc 3) (2) 승인키 요청 타임아웃
      (kis_auth.requests 래핑) (3) 30초 안에 ACK가 안 온 종목이 하나라도 있거나 장중 전 종목 무체결 120초면 재접속
      (ACK 누락 재접속은 연속 3회까지 - 이후엔 로그만) (4) 백오프는 체결 수신 후에만 초기화 + jitter (5) 이어쓰기 전
      반쪽 줄 개행 마감 (6) --until 기본값을 r014 특수일 마감+1분, 휴장일은 즉시 종료 (7) START 다음 줄에
      "CODES a,b,..."로 구독 요청 목록 기록(g024가 파일 없는 종목 대조).
    impact: collector (r003이 띄우는 g011 포함 - r003 재시작 후 반영)
    compatibility: backward-compatible (--until 명시 시 기존 동일, 출력 형식 불변)
- [2026-11-02] type=feat owner=claude
    summary: --fill-from-scan - r004 감시 종목 뒤에 가장 최근 g002 스캔(_<date>_scan_all.md)의 순위 순 후보를 40종목까지
      채워 구독(3중 트리거 재현/연구용 틱 확보). r004 종목이 항상 앞이므로 r003이 --codes 없는 g011을 재사용해도 감시
      종목은 모두 구독된다. g022_tick_collect_daily.sh가 r003과 별개로 매일 기동.
    impact: collector
    compatibility: backward-compatible (옵션 미지정 시 기존 동일)
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
import random
import signal
import sys
import time
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIVE_DIR = HERE.parent / "live_trading"
sys.path.insert(0, str(LIVE_DIR))

# [2026-11-10] --runtime DIR (r016 collector start; a process started through WMI on Windows inherits no environment) must be
# applied before r001 fixes RUNTIME_ROOT at import
if "--runtime" in sys.argv[:-1]:
    os.environ["AUTO_TRADING_RUNTIME"] = sys.argv[sys.argv.index("--runtime") + 1]

import r001_define_config as _cfg  # noqa: E402
from r001_define_config import RUNTIME_DATA_DIR, DEFINE_TODAY_CODE_PATH  # noqa: E402
from r009_tick_bars import TICK_FILE_COLUMNS  # noqa: E402
from r014_market_calendar import session_hours  # noqa: E402
import r019_platform as r019  # noqa: E402  (Linux/Windows locks)

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
# [2026-11-12] NXT executions: same record layout (one more trailing column), same websocket session
NXT_TR_ID = "H0NXCNT0"
VENUE_BY_TR = {TR_ID: "KRX", NXT_TR_ID: "NXT"}
NXT_SUBDIR = "nxt"                 # <runtime>/data/ticks/<day>/nxt/<code>.csv - never read by the KRX tick readers
SUBSCRIPTION_SYNC_SECONDS = 5.0    # how often the wanted subscriptions (by time of day) are compared with the active ones
LOCK_NAME = ".g011.lock"           # in <runtime>/data/ticks/ (date-independent: one collector per machine)
EXIT_ALREADY_RUNNING = 3
AUTH_TIMEOUT_SECONDS = (5.0, 10.0)  # (connect, read) for /oauth2/Approval - kis_auth sets no timeout
SUB_ACK_TIMEOUT_SECONDS = 30.0      # a requested code without any ACK (ok or error) after this -> reconnect
MAX_ACK_RECONNECTS = 3              # consecutive ACK-loss reconnects; afterwards keep the connection and only log
IDLE_RECONNECT_SECONDS = 120.0      # no execution for ANY code during continuous trading -> reconnect
IDLE_OPEN_GRACE_MINUTES = 2         # idle check starts open+2m ...
IDLE_CLOSE_AUCTION_MINUTES = 10     # ... and stops at close-10m (closing call auction prints only at the close)
MAX_BACKOFF_SECONDS = 60.0


class _Reconnect(Exception):
    """Raised inside the receive loop to drop a connection that is open but not delivering."""


class _RequestsWithTimeout:
    """Stand-in for kis_auth's `requests` that adds a timeout to calls without one (auth_ws has none, so a hung
    approval request would freeze the collector forever). Only the name inside kis_auth is replaced."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def post(self, url, *args, **kwargs):
        kwargs.setdefault("timeout", AUTH_TIMEOUT_SECONDS)
        return self._real.post(url, *args, **kwargs)

    def get(self, url, *args, **kwargs):
        kwargs.setdefault("timeout", AUTH_TIMEOUT_SECONDS)
        return self._real.get(url, *args, **kwargs)


def acquire_single_instance_lock(lock_dir: Path):
    """Open + exclusively flock the lock file; returns the open file (keep it for the process lifetime) or None
    when another collector holds it. Python fds are non-inheritable, so children never keep the lock."""
    return r019.acquire_instance_lock(lock_dir / LOCK_NAME)  # [2026-11-10] flock (Linux) / msvcrt (Windows)


def idle_check_window(day: datetime) -> tuple[datetime, datetime] | None:
    """[start, end) in which a feed with zero executions is considered dead, from the r014 calendar."""
    hours = session_hours(day)
    if hours is None:
        return None
    start = datetime.combine(day.date(), hours[0]) + timedelta(minutes=IDLE_OPEN_GRACE_MINUTES)
    end = datetime.combine(day.date(), hours[1]) - timedelta(minutes=IDLE_CLOSE_AUCTION_MINUTES)
    return start, end


VENUE = "KRX"   # H0STCNT0 = KRX executions only (NXT would be H0NXCNT0, unified H0UNCNT0)
CLOSE_AUCTION_WINDOW_MINUTES = 10
_DEFAULT_HOURS = (dt_time(9, 0), dt_time(15, 30))


def classify_session(trade_time: str, hours: tuple | None = None) -> str:
    """pre | regular | close_auction | post from a HHMMSS print time and the day's regular (open, close) hours.
    The 15:20-15:30 closing single-price auction has no prints; its fills print at/after the close time."""
    open_t, close_t = hours or _DEFAULT_HOURS
    try:
        hhmmss = str(trade_time).strip().zfill(6)
        t = dt_time(int(hhmmss[0:2]), int(hhmmss[2:4]), int(hhmmss[4:6]))
    except (TypeError, ValueError):
        return ""
    if t < open_t:
        return "pre"
    if t < close_t:
        return "regular"
    close_end = (datetime.combine(datetime(2000, 1, 1), close_t) + timedelta(minutes=CLOSE_AUCTION_WINDOW_MINUTES)).time()
    if t < close_end:
        return "close_auction"
    return "post"


def classify_nxt_session(trade_time: str, hours: tuple | None = None) -> str:
    """pre | regular | post for an NXT print: before the KRX open = pre-market, after the KRX close = after-market."""
    open_t, close_t = hours or _DEFAULT_HOURS
    try:
        hhmmss = str(trade_time).strip().zfill(6)
        t = dt_time(int(hhmmss[0:2]), int(hhmmss[2:4]), int(hhmmss[4:6]))
    except (TypeError, ValueError):
        return ""
    return "pre" if t < open_t else "regular" if t < close_t else "post"


def subscription_plan(now: datetime, krx_codes: list[str], nxt_codes: list[str], krx_until: datetime,
                      nxt_window: tuple[datetime, datetime] | None, cap: int = MAX_SUBSCRIPTIONS) -> list[tuple[str, str]]:
    """(tr_id, code) wanted at `now`. KRX first (until krx_until), NXT fills what is left of the session cap during its
    window, in watchlist order - outside KRX hours that is every NXT code."""
    out: list[tuple[str, str]] = []
    if now < krx_until:
        out += [(TR_ID, c) for c in krx_codes[:cap]]
    if nxt_window is not None and nxt_window[0] <= now < nxt_window[1]:
        out += [(NXT_TR_ID, c) for c in nxt_codes[:max(0, cap - len(out))]]
    return out


def parse_frame(raw: str, hours: tuple | None = None) -> list[dict] | None:
    """Split one data frame into records. Returns None for non-data (JSON control) frames.
    Format: <encrypt flag>|<tr_id>|<record count>|<f1^f2^...> (count x NCOL fields).
    hours = the day's regular (open, close) for the session column (None -> 09:00/15:30)."""
    if not raw or raw[0] not in "01":
        return None
    parts = raw.split("|", 3)
    if len(parts) < 4 or parts[1] not in VENUE_BY_TR:
        return []
    venue = VENUE_BY_TR[parts[1]]
    if parts[0] == "1":
        raise ValueError(f"encrypted {parts[1]} frame is not expected")
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
            "venue": venue,
            "session": (classify_session if venue == VENUE else classify_nxt_session)(rec[_IDX["STCK_CNTG_HOUR"]], hours),
        })
    return out


class TickWriter:
    """Append-only per-code CSV writer, flushed periodically. One file handle per code."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._files: dict[str, tuple] = {}
        self.rows = 0
        self.torn_rows = 0

    def write(self, recv_ts: str, rec: dict) -> None:
        code = str(rec["code"]).zfill(6)
        fh = self._files.get(code)
        if fh is None:
            path = self.out_dir / f"{code}.csv"
            new = not path.exists() or path.stat().st_size == 0
            torn = not new and not _ends_with_newline(path)
            fieldnames = TICK_FILE_COLUMNS if new else (_existing_header(path) or TICK_FILE_COLUMNS)
            f = open(path, "a", newline="", encoding="utf-8")
            if torn:
                f.write("\n")  # close the row cut by a crash; r009 drops it as a short row
                self.torn_rows += 1
            # an older (v1) file keeps its own width: readers drop rows whose width differs from the header
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
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


def _existing_header(path: Path) -> list[str] | None:
    """Column names of an existing tick file (first line), None if unreadable/empty."""
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            line = f.readline()
    except OSError:
        return None
    cols = next(csv.reader([line])) if line.strip() else []
    return cols or None


def _ends_with_newline(path: Path) -> bool:
    with open(path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        return f.read(1) == b"\n"


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


def scan_fill_codes(data_dir: Path, today: str) -> tuple[str, list[str]]:
    """(scan date, codes in rank order) of the newest g002 _<date>_scan_all.md dated <= today, else ("", [])."""
    dates = sorted((d.name for d in Path(data_dir).iterdir() if d.is_dir() and d.name.isdigit()
                    and len(d.name) == 8 and d.name <= today), reverse=True) if Path(data_dir).is_dir() else []
    for day in dates:
        path = Path(data_dir) / day / f"_{day}_scan_all.md"
        if not path.exists():
            continue
        codes = []
        for line in path.read_text(encoding="utf-8").splitlines():
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            # | rank | picked | code | name | ...
            if len(cols) >= 3 and cols[0].isdigit() and cols[2].isdigit() and len(cols[2]) == 6:
                codes.append(cols[2])
        return day, codes
    return "", []


async def run_collector(codes: list[str], until: datetime, out_dir: Path, log, nxt_codes: list[str] | None = None,
                        nxt_window: tuple[datetime, datetime] | None = None) -> None:
    """codes / until = the KRX subscriptions and their end (unchanged). nxt_codes + nxt_window add NXT executions on the
    same session (see subscription_plan); the collector then runs until the later of the two ends."""
    session_hours_today = session_hours(datetime.now()) or _DEFAULT_HOURS
    nxt_codes = list(nxt_codes or []) if nxt_window is not None else []
    end_all = max(until, nxt_window[1]) if nxt_codes else until
    nxt_failed: set = set()
    import websockets  # imported lazily so --dry-parse-test works without network deps

    open_api_root = Path(os.environ.get("OPEN_TRADING_API_ROOT", str(Path(__file__).resolve().parents[3] / "open-trading-api")))
    sys.path.insert(0, str(open_api_root / "examples_user"))
    sys.path.insert(0, str(open_api_root / "examples_user" / "domestic_stock"))
    import kis_auth as ka  # noqa: E402
    from domestic_stock_functions_ws import ccnl_krx  # noqa: E402
    ccnl_nxt = getattr(sys.modules.get("domestic_stock_functions_ws"), "ccnl_nxt", None)
    if nxt_codes and ccnl_nxt is None:
        log("WARN open-trading-api has no ccnl_nxt (H0NXCNT0) - NXT executions are not collected")
        nxt_codes, end_all = [], until
    sub_fn = {TR_ID: ccnl_krx, NXT_TR_ID: ccnl_nxt}

    if not isinstance(ka.requests, _RequestsWithTimeout):
        ka.requests = _RequestsWithTimeout(ka.requests)
    idle_window = idle_check_window(datetime.now())
    writer = TickWriter(out_dir)
    nxt_writer = TickWriter(out_dir / NXT_SUBDIR) if nxt_codes else None
    backoff = 1.0
    ack_reconnects = 0
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
        while not stop.is_set() and datetime.now() < end_all:
            try:
                ka.auth_ws()
                url = f"{ka.getTREnv().my_url_ws}/tryitout/{TR_ID}"
                async with websockets.connect(url, ping_interval=None, max_queue=4096) as ws:
                    active: set = set()

                    async def sync_subscriptions() -> None:
                        # [2026-11-12] subscribe / unsubscribe so the session holds what the time of day asks for
                        want = subscription_plan(datetime.now(), codes, [c for c in nxt_codes if c not in nxt_failed],
                                                 until, nxt_window)
                        want_set = set(want)
                        drop = sorted(active - want_set)
                        add = [k for k in want if k not in active]
                        for tr, code in drop:            # free the slots first (session cap)
                            msg, _cols = sub_fn[tr]("0", code)
                            await ws.send(json.dumps(msg))
                            active.discard((tr, code))
                            await asyncio.sleep(0.05)
                        for tr, code in add:
                            msg, _cols = sub_fn[tr]("1", code)
                            await ws.send(json.dumps(msg))
                            active.add((tr, code))
                            await asyncio.sleep(0.05)
                        if nxt_codes and (drop or add):
                            log(f"SUBSCRIBED krx={sum(tr == TR_ID for tr, _c in active)} "
                                f"nxt={sum(tr == NXT_TR_ID for tr, _c in active)} (+{len(add)} -{len(drop)})")

                    await sync_subscriptions()
                    krx_req = [c for tr, c in active if tr == TR_ID]   # the codes whose ACK is checked below
                    log(f"CONNECTED url={url} requested={len(active)}")
                    last_sync = time.monotonic()
                    sub_ok: set = set()
                    sub_fail: set = set()
                    sub_reported = False
                    ack_checked = False
                    got_data = False
                    connected_at = last_data = last_flush = time.monotonic()
                    while not stop.is_set() and datetime.now() < end_all:
                        mono = time.monotonic()
                        if nxt_codes and mono - last_sync >= SUBSCRIPTION_SYNC_SECONDS:
                            last_sync = mono
                            await sync_subscriptions()
                        if not ack_checked and mono - connected_at >= SUB_ACK_TIMEOUT_SECONDS:
                            ack_checked = True
                            missing = sorted(set(krx_req) - sub_ok - sub_fail)
                            if missing:
                                if ack_reconnects < MAX_ACK_RECONNECTS:
                                    ack_reconnects += 1
                                    raise _Reconnect(f"no subscribe ACK in {SUB_ACK_TIMEOUT_SECONDS:.0f}s for "
                                                     f"{len(missing)} code(s) {missing[:10]} (ok={len(sub_ok)} "
                                                     f"failed={len(sub_fail)}, try {ack_reconnects}/{MAX_ACK_RECONNECTS})")
                                log(f"WARN still no subscribe ACK for {len(missing)} code(s) {missing[:10]} after "
                                    f"{MAX_ACK_RECONNECTS} reconnects - keeping this connection")
                            else:
                                ack_reconnects = 0
                        now_dt = datetime.now()
                        if (idle_window is not None and idle_window[0] <= now_dt < idle_window[1] and now_dt < until
                                and mono - last_data >= IDLE_RECONNECT_SECONDS):
                            raise _Reconnect(f"IDLE no execution for {mono - last_data:.0f}s")
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        except asyncio.TimeoutError:
                            writer.flush()
                            if nxt_writer is not None:
                                nxt_writer.flush()
                            continue
                        recv_ts = datetime.now().isoformat(timespec="milliseconds")
                        recs = parse_frame(raw, session_hours_today)
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
                                if tr_id == NXT_TR_ID:
                                    # NXT answers stay out of the KRX ACK accounting; a refused code leaves the NXT list
                                    if body.get("rt_cd") not in (None, "0") and "UNSUBSCRIBE" not in str(body.get("msg1", "")).upper():
                                        nxt_failed.add(key)
                                        log(f"NXT SUBSCRIBE ERROR {key} {body.get('msg1')}")
                                    continue
                                if body.get("rt_cd") not in (None, "0"):
                                    sub_fail.add(key)
                                    log(f"SUBSCRIBE ERROR {key} {body.get('msg1')}")
                                elif body.get("rt_cd") == "0":
                                    sub_ok.add(key)
                                if krx_req and len(sub_ok) + len(sub_fail) >= len(krx_req) and not sub_reported:
                                    sub_reported = True
                                    ack_reconnects = 0  # every code answered (also when it arrives after the WARN)
                                    log(f"SUBSCRIPTIONS ok={len(sub_ok)} failed={len(sub_fail)} {sorted(sub_fail)}")
                            continue
                        if recs:
                            last_data = time.monotonic()
                            if not got_data:
                                got_data = True
                                backoff = 1.0  # reset only once the feed really delivers (not on a bare connect)
                        for rec in recs:
                            if rec["venue"] == VENUE:
                                writer.write(recv_ts, rec)
                            elif nxt_writer is not None:
                                nxt_writer.write(recv_ts, rec)
                        if time.monotonic() - last_flush >= 1.0:
                            writer.flush()
                            if nxt_writer is not None:
                                nxt_writer.flush()
                            last_flush = time.monotonic()
            except Exception as exc:  # network / auth errors: back off and reconnect (gap is visible via ACML_VOL)
                writer.flush()
                if nxt_writer is not None:
                    nxt_writer.flush()
                wait = backoff * random.uniform(1.0, 1.3)
                log(f"DISCONNECTED {type(exc).__name__}: {exc} | retry in {wait:.0f}s")
                try:
                    await asyncio.wait_for(stop.wait(), timeout=wait)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2.0, MAX_BACKOFF_SECONDS)
    finally:
        writer.close()
        if nxt_writer is not None:
            nxt_writer.close()
        log(f"STOPPED rows={writer.rows}" + (f" torn_rows_closed={writer.torn_rows}" if writer.torn_rows else "")
            + (f" nxt_rows={nxt_writer.rows}" if nxt_writer is not None else ""))


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
    ap.add_argument("--until", default=None,
                    help="HH:MM local time to stop (default: r014 calendar close + 1 min, i.e. 15:31 normally)")
    ap.add_argument("--dry-parse-test", action="store_true")
    ap.add_argument("--fill-from-scan", action="store_true",
                    help="after the watchlist codes, fill up to 40 with the newest g002 scan's ranked candidates")
    ap.add_argument("--runtime", default=None, help="runtime root (applied at import, before r001)")
    ap.add_argument("--no-nxt", action="store_true", help="KRX executions only (default: r001 TICK_COLLECT_NXT)")
    args = ap.parse_args()
    if args.dry_parse_test:
        _self_test()
        return
    codes = load_codes(args)
    watch_codes = list(codes)
    nxt_on = not args.no_nxt and bool(getattr(_cfg, "TICK_COLLECT_NXT", False))
    fill_note = ""
    # [2026-11-12] with NXT collection the scan fill leaves one slot per watchlist code for its NXT subscription
    fill_room = MAX_SUBSCRIPTIONS - len(codes) - (len(codes) if nxt_on else 0)
    if args.fill_from_scan and fill_room > 0:
        scan_day, ranked = scan_fill_codes(RUNTIME_DATA_DIR, datetime.now().strftime("%Y%m%d"))
        extra = [c for c in ranked if c not in codes][:fill_room]
        codes += extra
        fill_note = f" (watchlist {len(codes) - len(extra)} + scan {scan_day or 'none'} {len(extra)})"
    if not codes:
        print("no codes to subscribe")
        return
    now = datetime.now()
    if args.until:
        hh, mm = (int(x) for x in args.until.split(":"))
        until = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    else:
        hours = session_hours(now)
        if hours is None:
            print(f"{now:%Y-%m-%d %H:%M:%S} market closed today (r014 calendar) - not collecting", flush=True)
            return
        until = datetime.combine(now.date(), hours[1]) + timedelta(minutes=1)
    lock_fh = acquire_single_instance_lock(RUNTIME_DATA_DIR / "ticks")
    if lock_fh is None:
        print(f"{now:%Y-%m-%d %H:%M:%S} another g011 holds {RUNTIME_DATA_DIR / 'ticks' / LOCK_NAME} - exiting",
              flush=True)
        sys.exit(EXIT_ALREADY_RUNNING)
    out_dir = RUNTIME_DATA_DIR / "ticks" / now.strftime("%Y%m%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "_collector.log"

    def log(msg: str) -> None:
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"START codes={len(codes)}{fill_note} until={until:%H:%M} out={out_dir}")
    log("CODES " + ",".join(codes))
    # [2026-11-12] NXT executions of the watchlist codes (not the scan fill) during the NXT window
    nxt_codes, nxt_window = [], None
    if nxt_on:
        nxt_codes = watch_codes
        nxt_window = (datetime.combine(now.date(), getattr(_cfg, "MORNING_NXT_START", dt_time(8, 0))),
                      datetime.combine(now.date(), getattr(_cfg, "AFTERNOON_NXT_END", dt_time(20, 0))))
        log(f"NXT codes={len(nxt_codes)} window={nxt_window[0]:%H:%M}~{nxt_window[1]:%H:%M} out={out_dir / NXT_SUBDIR} "
            f"(slots left by KRX, watchlist order)")
    asyncio.run(run_collector(codes, until, out_dir, log, nxt_codes=nxt_codes, nxt_window=nxt_window))


if __name__ == "__main__":
    main()
