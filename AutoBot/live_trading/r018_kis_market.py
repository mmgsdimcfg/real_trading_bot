"""
r018_kis_market.py
AutoBotH 차트 탭용 KIS 시세 조회(읽기 전용) - r016 `quote` 명령과 r017 `bars`의 KIS 폴백(서버에 파일이 없을 때)이 쓴다.

지키는 것 (r003 실매매 보호)
- 시세 조회 TR 7개만 허용 목록(_TRS)으로 둔다. GET만 쓰고 주문·잔고·계좌 TR은 import도 하지 않는다(kis_auth/domestic_stock_functions
  자체를 import하지 않는다 - pandas/websockets 기동 비용도 없앤다).
- 토큰을 **발급하지 않는다.** ~/KIS/config/KIS<YYYYMMDD> 토큰 파일 중 유효한 것을 읽기만 한다(r003이 아침에 발급).
  없거나 만료면 KisError("token_not_ready") - kis_auth의 토큰 파일 쓰기는 원자적이지 않아 여러 프로세스가 auth()를 부르면
  빈 파일을 읽고 재발급(1분 1회 제한)으로 r003 인증이 실패할 수 있다.
- 프로세스 간 공용 호출 제한: <runtime>/cache/kis/.ratelimit 파일 락 + 최근 호출 시각, 초당 최대 RATE_MAX_PER_SEC건,
  호출 간격 RATE_MIN_GAP_SEC 이상. r003은 이 제한을 모르므로 이 기능의 몫을 작게 둔다(r003 ~1.6건/초 + 이 기능 <= 2건/초).
- EGW00201(초당 거래건수 초과)은 1초 뒤 1회만 재시도, 그래도 실패하면 KisError(호출자가 캐시를 쓰거나 rc 3).
- 토큰·앱키·앱시크릿은 출력/로그/캐시 어디에도 쓰지 않는다. 오류 메시지는 KIS msg_cd/msg1만 쓴다.

캐시 (<runtime>/cache/kis/)
- quote/<code>.json: 2초 안 재요청은 캐시(cached_ms 표시).
- daily/<code>.json: 최근 3년 일봉. 같은 날 15:40 이후 받은 것은 그날 끝까지, 장중에 받은 것은 10분 유효.
- min/<YYYYMMDD>/<code>.json: 1분봉(t = 봉 시작). 지난 날짜는 다 받으면(complete) 다시 받지 않는다. 당일은 마지막
  미완성 분부터만 다시 받는다(보통 1회 호출). 세션 시각은 r014 달력, 휴장일·장 시작 전은 None.
- 마감 시각: 명령 하나(프로세스)에 Client 하나(shared_client), 모든 조회가 DEADLINE_SEC 를 나눠 쓴다. 남은 시간이 없으면
  KisError("deadline"). 캐시 쓰기 실패는 무시(최선 노력), 제한 파일 실패는 호출하지 않고 KisError("rate_storage").
- names.json: 종목명/시장(search_stock_info, 종목당 1회).

Update log:
- [2026-11-07] type=feat owner=claude
    summary: 신규 - AutoBotH 요청(server_kis_market_data_request_20261106) 읽기 전용 시세 조회. 토큰 미발급, 공용 호출 제한,
      캐시, 허용 TR 목록.
    impact: tooling (r003 코드 변경 없음)
    compatibility: backward-compatible
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
import yaml

import r019_platform as r019  # [2026-11-10] Linux/Windows file locks

CODE_RE = re.compile(r"^[0-9A-Z]{6}$")
RATE_MAX_PER_SEC = 2
RATE_MIN_GAP_SEC = 0.3
HTTP_TIMEOUT = (3.05, 7.0)
QUOTE_CACHE_SEC = 2.0
DAILY_CACHE_SEC = 600.0
DAILY_FINAL_AFTER = "18:00"  # KIS 일봉 당일 행은 16:38 이후에도 바뀐 사례가 있다(g001 노트)
DAILY_YEARS = 3
MIN_TODAY_CACHE_SEC = 5.0
DEADLINE_SEC = 25.0  # 명령 하나 전체(모든 TR 호출·페이징·예열 합계) - r016/AutoBotH 타임아웃 30초 안
MARKET = "J"  # KRX (틱 수집기·r003과 같은 기준). UN/NX는 쓰지 않는다.

# name -> (path, tr_id). 이 목록 밖의 TR은 _get이 거부한다 (주문/잔고/계좌 TR 없음).
_TRS = {
    "inquire_price": ("/uapi/domestic-stock/v1/quotations/inquire-price", "FHKST01010100"),
    "inquire_asking_price_exp_ccn": ("/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn", "FHKST01010200"),
    "inquire_ccnl": ("/uapi/domestic-stock/v1/quotations/inquire-ccnl", "FHKST01010300"),
    "inquire_time_itemchartprice": ("/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice", "FHKST03010200"),
    "inquire_time_dailychartprice": ("/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice", "FHKST03010230"),
    "inquire_daily_itemchartprice": ("/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice", "FHKST03010100"),
    "search_stock_info": ("/uapi/domestic-stock/v1/quotations/search-stock-info", "CTPF1002R"),
}


class KisError(Exception):
    """KIS 오류 또는 사용 불가. code: KIS msg_cd 또는 token_not_ready / rate_busy / http_<status> / network."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# paths / config (secrets stay in memory only)
# ---------------------------------------------------------------------------
def _kis_config_dir() -> Path:
    return Path(os.environ.get("KIS_CONFIG_DIR") or Path.home() / "KIS" / "config")


def _runtime_root() -> Path:
    return Path(os.environ.get("AUTO_TRADING_RUNTIME") or Path(__file__).resolve().parents[3] / "runtime")


def _cache_dir() -> Path:
    return _runtime_root() / "cache" / "kis"


def _now() -> datetime:
    return datetime.now()


def load_token(now: datetime | None = None) -> str:
    """가장 최근의 유효한 토큰 파일(KIS<YYYYMMDD>)의 토큰. 발급하지 않는다."""
    now = now or _now()
    files = sorted((p for p in _kis_config_dir().glob("KIS*") if re.fullmatch(r"KIS\d{8}", p.name)), key=lambda p: p.name, reverse=True)
    for p in files:
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict) or not data.get("token"):
            continue
        valid = data.get("valid-date")
        if isinstance(valid, str):
            try:
                valid = datetime.strptime(valid.strip(), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
        if isinstance(valid, datetime) and valid > now + timedelta(seconds=60):
            return str(data["token"])
    raise KisError("token_not_ready", "no valid KIS token file (r003 issues it when it starts; this command never issues tokens)")


def _load_app() -> dict:
    p = _kis_config_dir() / "kis_devlp.yaml"
    try:
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        return {"url": str(cfg["prod"]).rstrip("/"), "appkey": cfg["my_app"], "appsecret": cfg["my_sec"],
                "agent": cfg.get("my_agent") or "Mozilla/5.0"}
    except Exception as exc:  # 내용은 출력하지 않는다
        raise KisError("config", f"cannot read kis_devlp.yaml ({type(exc).__name__})") from None


# ---------------------------------------------------------------------------
# shared rate limit (all processes using this module)
# ---------------------------------------------------------------------------
def _rate_slot(deadline: float) -> None:
    try:
        _rate_slot_inner(deadline)
    except OSError as exc:  # 제한 파일을 못 쓰면 호출하지 않는다(제한 우회 금지)
        raise KisError("rate_storage", f"rate limiter file unavailable ({type(exc).__name__})") from None


def _rate_slot_inner(deadline: float) -> None:
    d = _cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / ".ratelimit"
    with open(path, "a+", encoding="utf-8") as fh:
        while not r019.try_lock(fh):
            if time.time() > deadline:
                raise KisError("rate_busy", "shared KIS rate limiter busy") from None
            time.sleep(0.05)
        try:
            fh.seek(0)
            try:
                stamps = [float(x) for x in json.loads(fh.read() or "[]")]
            except (ValueError, TypeError):
                stamps = []
            while True:
                now = time.time()
                stamps = [s for s in stamps if now - s < 1.0]
                wait = 0.0
                if stamps:
                    wait = max(wait, RATE_MIN_GAP_SEC - (now - max(stamps)))
                if len(stamps) >= RATE_MAX_PER_SEC:
                    wait = max(wait, 1.0 - (now - min(stamps)))
                if wait <= 0:
                    break
                if now + wait > deadline:
                    raise KisError("rate_busy", "shared KIS rate limit (would exceed deadline)")
                time.sleep(wait)
            stamps.append(time.time())
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps(stamps))
            fh.flush()
        finally:
            r019.unlock(fh)


class Client:
    """한 명령 실행 동안 쓰는 읽기 전용 클라이언트. calls = 실제 HTTP 호출 수."""

    def __init__(self, deadline_sec: float = DEADLINE_SEC, session=None):
        self.deadline = time.time() + deadline_sec
        self.calls = 0
        self._session = session or requests
        self._headers: dict | None = None
        self._url = ""

    def _base(self) -> dict:
        if self._headers is None:
            token = load_token()
            app = _load_app()
            self._url = app["url"]
            self._headers = {"content-type": "application/json; charset=utf-8", "authorization": f"Bearer {token}",
                             "appkey": app["appkey"], "appsecret": app["appsecret"], "custtype": "P",
                             "User-Agent": app["agent"]}
        return self._headers

    def time_left(self) -> float:
        return self.deadline - time.time()

    def get(self, tr: str, params: dict, tr_cont: str = "") -> tuple[dict, str]:
        if tr not in _TRS:
            raise ValueError(f"TR not allowed: {tr}")
        path, tr_id = _TRS[tr]
        headers = dict(self._base(), tr_id=tr_id, tr_cont=tr_cont)
        for attempt in (0, 1):
            if self.time_left() < 1.0:
                raise KisError("deadline", "command time budget used up")
            _rate_slot(self.deadline - 0.5)
            left = self.time_left()
            if left < 0.5:
                raise KisError("deadline", "command time budget used up")
            timeout = (min(HTTP_TIMEOUT[0], left), min(HTTP_TIMEOUT[1], left))
            self.calls += 1
            try:
                res = self._session.get(self._url + path, headers=headers, params=params, timeout=timeout)
            except requests.Timeout as exc:
                raise KisError("timeout", type(exc).__name__) from None
            except requests.RequestException as exc:
                raise KisError("network", type(exc).__name__) from None
            try:
                body = res.json()
            except ValueError:
                body = {}
            if not isinstance(body, dict):
                body = {}
            msg_cd = str(body.get("msg_cd") or "")
            if res.status_code == 200 and str(body.get("rt_cd")) == "0":
                return body, str(res.headers.get("tr_cont", "") or "")
            if msg_cd == "EGW00201" and attempt == 0 and self.time_left() > 1.5:
                time.sleep(1.0)
                continue
            raise KisError(msg_cd or f"http_{res.status_code}", str(body.get("msg1") or "").strip() or f"HTTP {res.status_code}")
        raise KisError("EGW00201", "rate limit")  # pragma: no cover


_SHARED: Client | None = None
# 페이징 도중 이 오류면 받은 페이지까지는 응답·저장한다(완료 표시는 하지 않음)
_PARTIAL_OK = ("deadline", "rate_busy", "timeout")


def shared_client() -> Client:
    """명령(프로세스) 하나에 클라이언트 하나 - 마감 시각 DEADLINE_SEC 를 모든 조회(본 요청 + 예열)가 나눠 쓴다."""
    global _SHARED
    if _SHARED is None:
        _SHARED = Client()
    return _SHARED


# ---------------------------------------------------------------------------
# cache helpers
# ---------------------------------------------------------------------------
def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path: Path, obj) -> None:
    """원자적 캐시 쓰기. 실패해도(디스크 가득/권한) 조회 결과는 버리지 않는다 - 캐시는 최선 노력."""
    tmp = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            out.write(json.dumps(obj, ensure_ascii=False))
        os.replace(tmp, path)
        tmp = None
    except OSError:
        pass
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _f(v) -> float | None:
    try:
        x = float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return int(x) if x.is_integer() else x


def _check_code(code: str) -> str:
    code = str(code).strip().upper()
    if not CODE_RE.fullmatch(code):
        raise ValueError(f"invalid code {code!r}")
    return code


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------
_MARKETS = {"STK": "KOSPI", "KSQ": "KOSDAQ", "KNX": "KONEX"}


def name_info(code: str, client: Client | None = None) -> dict:
    """{"name", "market"} - names.json 캐시, 없으면 search_stock_info 1회. 실패하면 빈 값."""
    path = _cache_dir() / "names.json"
    names = _read_json(path) or {}
    if code in names and names[code].get("name"):
        return names[code]
    if client is None:
        return {"name": None, "market": None}
    try:
        body, _ = client.get("search_stock_info", {"PRDT_TYPE_CD": "300", "PDNO": code})
    except KisError:
        return {"name": None, "market": None}
    out = body.get("output") or {}
    info = {"name": (out.get("prdt_abrv_name") or out.get("prdt_name") or "").strip() or None,
            "market": _MARKETS.get(str(out.get("mket_id_cd") or ""), out.get("mket_id_cd") or None)}
    if info["name"]:
        names = _read_json(path) or {}
        names[code] = info
        _write_json(path, names)
    return info


def remember_name(code: str, name: str | None) -> None:
    if not name:
        return
    path = _cache_dir() / "names.json"
    names = _read_json(path) or {}
    if names.get(code, {}).get("name") != name:
        names[code] = {"name": name, "market": names.get(code, {}).get("market")}
        _write_json(path, names)


# ---------------------------------------------------------------------------
# quote
# ---------------------------------------------------------------------------
def _hms(s) -> str:
    s = str(s or "").strip().zfill(6)
    return f"{s[0:2]}:{s[2:4]}:{s[4:6]}" if s.isdigit() else ""


def quote(code: str, trades: int = 30, client: Client | None = None) -> dict:
    code = _check_code(code)
    trades = max(0, min(int(trades), 30))
    cpath = _cache_dir() / "quote" / f"{code}.json"
    cached = _read_json(cpath)
    if cached and time.time() - float(cached.get("_at", 0)) < QUOTE_CACHE_SEC:
        out = {k: v for k, v in cached.items() if k != "_at"}
        out["cached_ms"] = int((time.time() - float(cached["_at"])) * 1000)
        out["trades"] = (out.get("trades") or [])[:trades]
        return out
    client = client or shared_client()
    params = {"FID_COND_MRKT_DIV_CODE": MARKET, "FID_INPUT_ISCD": code}
    body, _ = client.get("inquire_price", params)
    p = body.get("output") or {}
    price = _f(p.get("stck_prpr"))
    if not price:
        raise ValueError(f"unknown code {code}")
    info = name_info(code, client)
    out = {
        "ok": True, "code": code, "name": info.get("name"), "market": info.get("market"),
        "asof": _now().isoformat(timespec="seconds"), "source": "kis", "cached_ms": 0,
        "price": price, "change": _f(p.get("prdy_vrss")), "change_pct": _f(p.get("prdy_ctrt")),
        "prev_close": _f(p.get("stck_sdpr")), "base_price": _f(p.get("stck_sdpr")),
        "open": _f(p.get("stck_oprc")), "high": _f(p.get("stck_hgpr")), "low": _f(p.get("stck_lwpr")),
        "volume": _f(p.get("acml_vol")), "amount": _f(p.get("acml_tr_pbmn")),
        "upper_limit": _f(p.get("stck_mxpr")), "lower_limit": _f(p.get("stck_llam")),
        "w52_high": _f(p.get("w52_hgpr")), "w52_high_date": p.get("w52_hgpr_date") or None,
        "w52_low": _f(p.get("w52_lwpr")), "w52_low_date": p.get("w52_lwpr_date") or None,
        "per": _f(p.get("per")), "pbr": _f(p.get("pbr")), "market_cap_eok": _f(p.get("hts_avls")),
        "vi": str(p.get("vi_cls_code") or "N").strip() not in ("", "N", "0"),
        "halted": str(p.get("temp_stop_yn") or "N").strip().upper() == "Y",
        "strength": None, "orderbook": None, "trades": [], "partial_errors": [],
    }
    try:
        if client.time_left() < 2:
            raise KisError("deadline", "skipped (time budget)")
        body, _ = client.get("inquire_asking_price_exp_ccn", params)
        b = body.get("output1") or {}
        asks = [[_f(b.get(f"askp{i}")), _f(b.get(f"askp_rsqn{i}"))] for i in range(1, 11)]
        bids = [[_f(b.get(f"bidp{i}")), _f(b.get(f"bidp_rsqn{i}"))] for i in range(1, 11)]
        out["orderbook"] = {"asof": _hms(b.get("aspr_acpt_hour")), "asks": [a for a in asks if a[0]], "bids": [x for x in bids if x[0]],
                            "total_ask": _f(b.get("total_askp_rsqn")), "total_bid": _f(b.get("total_bidp_rsqn"))}
    except KisError as exc:
        out["partial_errors"].append(f"orderbook {exc.code}: {exc.message}")
    try:
        if client.time_left() < 2:
            raise KisError("deadline", "skipped (time budget)")
        body, _ = client.get("inquire_ccnl", params)
        rows = body.get("output") or []
        if rows:
            out["strength"] = _f(rows[0].get("tday_rltv"))
        # inquire_ccnl 에는 매수/매도 체결 구분이 없다 -> "" (모름)
        out["trades"] = [[_hms(r.get("stck_cntg_hour")), _f(r.get("stck_prpr")), _f(r.get("cntg_vol")), ""] for r in rows[:30]]
    except KisError as exc:
        out["partial_errors"].append(f"trades {exc.code}: {exc.message}")
    _write_json(cpath, dict(out, _at=time.time()))
    out["trades"] = out["trades"][:trades]
    return out


# ---------------------------------------------------------------------------
# daily (D) bars - last DAILY_YEARS years
# ---------------------------------------------------------------------------
def _daily_fresh(fetched_at: float, now: datetime) -> bool:
    f = datetime.fromtimestamp(fetched_at)
    if f.date() != now.date():
        return False
    final = datetime.combine(now.date(), datetime.strptime(DAILY_FINAL_AFTER, "%H:%M").time())
    if f >= final:
        return True
    return (now - f).total_seconds() < DAILY_CACHE_SEC


def daily(code: str, client: Client | None = None) -> dict:
    """{"rows": [[YYYYMMDD, o, h, l, c, v], ...] 오름차순, "name", "source", "calls"}."""
    code = _check_code(code)
    now = _now()
    cpath = _cache_dir() / "daily" / f"{code}.json"
    cached = _read_json(cpath)
    if cached and cached.get("complete") is True and _daily_fresh(float(cached.get("fetched_at", 0)), now):
        return {"rows": cached["rows"], "name": cached.get("name"), "source": "kis:inquire_daily_itemchartprice", "calls": 0,
                "complete": True}
    client = client or shared_client()
    start = (now - timedelta(days=365 * DAILY_YEARS + 3)).strftime("%Y%m%d")
    end = now.strftime("%Y%m%d")
    rows: dict[str, list] = {}
    name = None
    exhausted = False
    for _page in range(12):
        try:
            body, _ = client.get("inquire_daily_itemchartprice", {
                "FID_COND_MRKT_DIV_CODE": MARKET, "FID_INPUT_ISCD": code, "FID_INPUT_DATE_1": start,
                "FID_INPUT_DATE_2": end, "FID_PERIOD_DIV_CODE": "D", "FID_ORG_ADJ_PRC": "0"})
        except KisError as exc:
            if rows and exc.code in _PARTIAL_OK:
                break  # 받은 만큼 응답, complete=False 로 저장 -> 다음 요청이 이어 받는다
            raise
        name = name or ((body.get("output1") or {}).get("hts_kor_isnm") or "").strip() or None
        page = [r for r in (body.get("output2") or []) if str(r.get("stck_bsop_date") or "").strip()]
        new = 0
        for r in page:
            d = str(r["stck_bsop_date"]).strip()
            o, h, l, c = (_f(r.get(k)) for k in ("stck_oprc", "stck_hgpr", "stck_lwpr", "stck_clpr"))
            if not c:
                continue
            if d not in rows:
                new += 1
            rows[d] = [d, o, h, l, c, _f(r.get("acml_vol"))]
        if not page or new == 0:
            exhausted = True
            break
        oldest = min(str(r["stck_bsop_date"]).strip() for r in page)
        if oldest <= start or len(page) < 100:
            exhausted = True
            break
        if client.time_left() < 3:
            break
        end = (datetime.strptime(oldest, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
    if not rows:
        raise ValueError(f"unknown code {code} (KIS returned no daily rows)")
    out_rows = [rows[k] for k in sorted(rows)]
    # 시간 제한으로 3년을 다 못 받았으면 complete=False 로 저장 -> 다음 요청이 다시 받는다(그동안 받은 만큼은 응답)
    _write_json(cpath, {"fetched_at": time.time(), "rows": out_rows, "name": name, "complete": exhausted})
    remember_name(code, name)
    return {"rows": out_rows, "name": name, "source": "kis:inquire_daily_itemchartprice", "calls": client.calls,
            "complete": exhausted}


# ---------------------------------------------------------------------------
# 1-minute bars (t = bar START, like the 10s files / ticks)
# ---------------------------------------------------------------------------
def _minus_minute(hms: str) -> str:
    t = datetime.strptime(hms, "%H%M%S") - timedelta(minutes=1)
    return t.strftime("%H%M%S")


def _parse_min_rows(rows: list, date_str: str) -> dict[str, list]:
    out = {}
    for r in rows:
        if str(r.get("stck_bsop_date") or "").strip() != date_str:
            continue  # FID_PW_DATA_INCU_YN=Y 는 전일 분봉도 섞어 준다
        hms = str(r.get("stck_cntg_hour") or "").strip().zfill(6)
        o, h, l, c = (_f(r.get(k)) for k in ("stck_oprc", "stck_hgpr", "stck_lwpr", "stck_prpr"))
        if not hms.isdigit() or not c:
            continue
        out[hms] = [hms, o, h, l, c, _f(r.get("cntg_vol"))]
    return out


def _session_hours(date_str: str) -> tuple[str, str] | None:
    """(첫 분봉 HHMMSS, 마지막 분봉 HHMMSS) - r014 달력(수능일 10:00~16:30 등). 마지막 = 종가 동시호가 분(close)."""
    import r014_market_calendar as r014  # noqa: E402  (가벼움)
    try:
        h = r014.session_hours(date_str)
    except Exception:
        h = None
        try:
            if datetime.strptime(date_str, "%Y%m%d").weekday() < 5:
                return "090000", "153000"
        except ValueError:
            return None
    if h is None:
        return None
    return h[0].strftime("%H%M%S"), h[1].strftime("%H%M%S")


def minutes(code: str, date_str: str, client: Client | None = None) -> dict | None:
    """{"rows": [[HHMMSS, o, h, l, c, v], ...] 오름차순, "name", "source", "complete", "calls"} 또는 None(KIS에 없음)."""
    code = _check_code(code)
    now = _now()
    today = now.strftime("%Y%m%d")
    if date_str > today:
        return None
    is_today = date_str == today
    hours = _session_hours(date_str)
    if hours is None:
        return None  # 휴장일(r014 달력)
    open_hms, last_hms = hours
    if is_today and now.strftime("%H%M%S") < open_hms:
        return None  # 장 시작 전 - 오늘 분봉 없음
    tr = "inquire_time_itemchartprice" if is_today else "inquire_time_dailychartprice"
    cpath = _cache_dir() / "min" / date_str / f"{code}.json"
    cached = _read_json(cpath) or {}
    have: dict[str, list] = {r[0]: r for r in cached.get("rows") or []}
    if cached.get("complete") and not is_today:
        return {"rows": cached["rows"], "name": cached.get("name"), "source": f"kis:{tr}", "complete": True, "calls": 0}
    if is_today and have and time.time() - float(cached.get("fetched_at", 0)) < MIN_TODAY_CACHE_SEC:
        return {"rows": cached["rows"], "name": cached.get("name"), "source": f"kis:{tr}", "complete": False, "calls": 0}
    client = client or shared_client()
    name = cached.get("name")
    # 09:00부터 끊김 없이 받아 둔 경우에만 증분: 마지막 분은 미완성일 수 있어 다시 받고, 그 직전 분까지 겹치면 멈춘다.
    # (앞선 조회가 시간 제한으로 09:00까지 못 갔으면 처음부터 다시 받아 구멍을 남기지 않는다)
    keep_until = _minus_minute(max(have)) if (is_today and have and cached.get("complete_from_open")) else None
    hour = min(now.strftime("%H%M%S"), last_hms) if is_today else last_hms
    reached_open = False
    overlapped = False
    fetched_any = False
    for _page in range(16):
        params = {"FID_COND_MRKT_DIV_CODE": MARKET, "FID_INPUT_ISCD": code, "FID_INPUT_HOUR_1": hour,
                  "FID_PW_DATA_INCU_YN": "Y"}
        if is_today:
            params["FID_ETC_CLS_CODE"] = ""
        else:
            params.update({"FID_INPUT_DATE_1": date_str, "FID_FAKE_TICK_INCU_YN": ""})
        try:
            body, _ = client.get(tr, params)
        except KisError as exc:
            if fetched_any and exc.code in _PARTIAL_OK:
                break  # 받은 만큼 저장(끊김 없음 표시는 하지 않음) -> 다음 조회가 메운다
            raise
        name = name or ((body.get("output1") or {}).get("hts_kor_isnm") or "").strip() or None
        page = _parse_min_rows(body.get("output2") or [], date_str)
        if not page:
            # 이번 조회에서 최신부터 이어 받은 뒤에 빈 페이지면 더 이전 분봉이 없는 것(장 시작 쪽 끝).
            # 첫 페이지부터 비었으면 연속성을 확인한 것이 아니므로 표시를 바꾸지 않는다.
            reached_open = fetched_any
            break
        fetched_any = True
        have.update(page)
        oldest = min(page)
        if oldest <= open_hms:
            reached_open = True
            break
        if keep_until and oldest <= keep_until:
            overlapped = True
            break
        if client.time_left() < 2:
            break
        hour = _minus_minute(oldest)
    if not have:
        return None
    rows = [have[k] for k in sorted(have)]
    complete = (not is_today) and reached_open
    # 이번 조회가 장 시작까지 갔거나, 앞서 09:00부터 이어 받은 구간과 실제로 겹쳤을 때만 '끊김 없음' 유지.
    # 시간 제한으로 중간에 멈추면 False -> 다음 조회가 처음부터 다시 받아 구멍을 메운다.
    from_open = reached_open or overlapped
    _write_json(cpath, {"fetched_at": time.time(), "rows": rows, "name": name, "complete": complete,
                        "complete_from_open": from_open})
    remember_name(code, name)
    return {"rows": rows, "name": name, "source": f"kis:{tr}", "complete": complete, "calls": client.calls}

