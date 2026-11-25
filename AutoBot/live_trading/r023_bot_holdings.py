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
- [2026-11-25] type=feat owner=claude
    summary: 사용자 결정(체결 내역 기준 장부, Codex 계획 검토, r023_fill_ledger_plan_20261125.md) - 체결 장부 bot_fill_ledger.json(schema 2):
      봇 주문 등록부(registry: 주문일:주문번호 -> 종목·방향·수량·종결 여부, 원주문번호로 이어진 정정 주문 포함) + 반영 체결(applied) + 종목별
      봇 수량(codes). 봇 몫은 등록부 주문의 체결로만 바뀌고(apply_fill_rows), 계좌 수량은 상한(reconcile_account: 직접 매도는 직접 몫 먼저 -
      계좌가 봇 몫보다 작아질 때만 봇 몫이 준다, 봇 주문 진행 중·체결 직후 정착 시간에는 자르지 않음). 팔 수 있는 봇 수량 = 봇 몫 - 아직 반영
      안 된 봇 매도 주문의 남은 수량(reserved_sell). 기존 bot_holdings.json 은 그대로 두고 첫 전환 때 봇 수량을 이어받음(사용자 결정).
      update_ledger / seed_ledger(수량 차이 방식)는 호환용으로 남김.
      Codex 코드 검토 반영: 계좌 상한은 종목 전체를 건너뛰지 않고 "최근 정착 시간 안에 반영한 봇 매수 체결량"만 여유로 둠
      (settle_allowance - 미체결 봇 매수가 수동 매도를 가리지 않게), 기준점(baseline) 주문은 첫 완전 조회의 누적 체결을 이미 반영된 것으로.
      Codex 재검토 반영: 정착 여유는 정착 시간 안에 반영한 체결 증분만(recent), 정정 주문은 원주문의 기준점을 이어받음, 전환 중 표시
      (migrating)를 파일에 보존.
      Codex 3차 반영: 계좌 상한 여유 = 최근 봇 매수 체결 증분 + 아직 반영 안 된 봇 매도 주문 잔량(reconcile_room - 잔고가 체결 조회보다
      먼저 줄어도 같은 매도를 두 번 빼지 않음), 주문번호 없는 매도 차감은 키로 한 번만(deduct_once, state["deducted"]).
      Codex 4차 반영: 매도 잔량 여유 대신 "선반영"(pre_applied) - 계좌가 봇 몫보다 작아지면 바로 줄이고, 줄인 양을 진행 중인 봇 매도
      주문에 선반영으로 달아 두었다가 그 체결이 오면 선반영만큼은 다시 빼지 않음(매도가 취소되면 봇 몫이 작게 남음 - 덜 파는 쪽).
      기준점은 맞는 행이 보일 때까지 유지(지난 날짜인데 안 보이면 등록 삭제).
      Codex 5차 후 단순화(추측 없는 보수 규칙): 선반영(pre_applied)·시간 유예(recent) 제거. 계좌가 봇 몫보다 작으면 즉시 줄이고(이후 봇
      매도 체결이 또 빠지면 봇 몫이 작아질 뿐 - 덜 파는 쪽), 여유는 호출자가 넘기는 "이번 조회에서 새로 본 봇 매수 체결량"만
      (계좌 스냅샷이 그보다 앞설 수 있는 양).
      Codex 6차 후 재설계(v3, 추측 제거): 주문체결 조회에는 MTS 직접 주문도 나온다 - 등록부에 없는 주문의 체결은 직접 몫(manual_qty)으로
      반영(직접 매도는 직접 몫 먼저, 모자라면 봇 몫; 봇 매도는 봇 몫 먼저, 모자라면 직접 몫). 봇·직접 몫이 모두 체결로만 바뀐다.
      계좌 수량은 안정된 종목(진행 중 주문 없음, 이번 조회에서 체결 변화 없음, 이번 차례에 조회가 실제로 돈 경우)에서만 맞춰 보며 차이는
      직접 몫으로(모자라면 봇 몫을 줄임 - 봇 몫은 늘리지 않음). 직접 몫을 모르면(전환 직후) 그 계좌 수량 - 봇 몫으로 시작.
      시간 유예·여유 함수(settle_allowance / reconcile_room / fresh_buy_room / settling_codes)는 제거.
      Codex 7차 반영: 한 조회의 새 체결은 매도를 모두 먼저, 매수를 나중에 반영(체결 순서를 모를 때 봇 몫이 가장 작게 나오는 순서),
      직접 몫을 모르는 종목의 초기화도 안정된 때에만.
      Codex 8차 반영: 매도를 먼저 반영할 때 몫이 모자라 다 빼지 못한 양은 같은 조회의 매수에서 갚음(봇 매수부터 - 봇 몫이 가장 작게),
      다시 조회하는 날짜(미종결 주문의 주문일 이후)의 반영 기록은 7일이 지나도 지우지 않음.
      Codex 9차 반영: 마지막으로 다 읽은 조회일(listed_through)을 저장해 날짜가 바뀐 뒤 첫 조회는 그날부터(전날 마지막 조회 뒤의
      직접 체결도 반영), 계좌 대사로 직접 몫에 먼저 더한 양(pending_up)은 나중에 조회에 나온 직접 매수와 상계(두 번 더하지 않음).
      Codex 10차 반영: pending_up 은 직접 몫과 따로 {qty, date} 로 두고(직접 매도로 줄지 않음) 그 날짜가 조회 범위 밖이 되면 만료,
      전날 체결은 오늘 하루 동안 계속 조회 범위에 포함(listed_prev). 계좌가 먼저 보여 준 직접 매도는 상계하지 않음(봇 몫이 작아지는
      쪽 - 그만큼 자동 청산에서 빠짐, 직접 보유분을 팔지는 않음).
      Codex 11차 반영: 매도(직접·봇 초과분)는 확정된 직접 몫(직접 몫 - pending_up)에서만 차감 - 대사로 먼저 더한 잠정 수량이 그보다
      앞선 매도의 재원이 되지 않음(순서를 모르면 봇 몫이 작아지는 쪽).
      Codex 12차 반영(순서와 무관한 상한): 그날 첫 조회 직전의 장부를 기준점(day_anchor)으로 두고, 조회마다 그날 체결 전체(누적)를
      "매도 먼저" 순서로 기준점에 적용한 봇 몫을 상한으로 삼아 장부의 봇 몫이 더 크면 차이를 직접 몫으로 옮김(clamp_to_day_bound) -
      체결이 여러 조회에 어떤 순서로 나뉘어 보여도 봇 몫은 가능한 가장 작은 값 이하. 체결 시각이 없어 같은 날 봇 보유 종목을 직접
      샀다 팔면(왕복) 그만큼 봇 몫이 직접 몫으로 넘어감. 직접 몫 초기화 값은 pending_up 에 넣지 않음(직접 매도가 봇 몫을 깎지 않게).
      Codex 13차 반영: 직접 몫 초기화는 같은 계좌 수량이 confirm_seconds 간격의 두 안정 조회에서 확인된 뒤에만(계좌에만 먼저 보인
      직접 매수는 그 사이 조회에 나와 반영 기록에 남음 - 이중 반영 없음), 상한 계산의 기준 체결량은 max(anchor.applied, base) 한 번만.
      Codex 14차 반영: 직접 몫을 확정하는 순간 그 종목의 기준점을 다시 잡음(그때의 봇·직접 수량 + 그날 주문별 누적 반영량) -
      그날 나머지 체결에도 상한이 적용됨.
      Codex 15차 반영: day_fill_totals - 조회의 그날 종목별 매수/매도 체결 합계(r003 이 계좌 잔고의 금일 매수/매도 수량과 대조해
      조회가 계좌보다 늦으면 bot_only 매도를 보류).
      Codex 17차 반영: 다 빼지 못한 매도량(debt)을 장부에 저장해 다음 조회의 매수(봇 매수부터)에서도 갚음 - 그 종목이 계좌와
      안정적으로 맞춰지면(reconcile 의 stable) 지움. 기준점에도 그때의 debt 를 함께 저장.
    impact: live (r003 의 bot_only 자동매도 수량)
    compatibility: 새 파일. 기존 파일은 읽기만(백업 후)
- [2026-11-25] type=feat owner=claude
    summary: update_ledger(buy_caps, sell_caps) - r003 주문 보류(과제 A2)로만 아는 종목은 그 주문 수량까지만 봇 수량으로 귀속(초과분은 직접 매매).
    impact: live (r003 재시작 직후 보류 종목의 장부)
    compatibility: backward-compatible (caps 없으면 종전과 동일)
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
                  bot_buy_codes=frozenset(), bot_sell_codes=frozenset(),
                  buy_caps: dict[str, int] | None = None, sell_caps: dict[str, int] | None = None) -> dict[str, dict]:
    """New ledger after one account sync (pure). account_qty: {code: quantity now in the account} (only held symbols).
    bot_buy_codes / bot_sell_codes: symbols for which the bot had a buy / sell order working since the previous sync.
    [2026-11-25] buy_caps / sell_caps: {code: most shares a working bot order can still explain} for symbols whose order is only
    known from an r003 order hold (r003 restarted while it was open) - a change beyond it is manual (bought / sold by hand)."""
    buy_caps = {_norm(c): max(0, int(v)) for c, v in (buy_caps or {}).items()}
    sell_caps = {_norm(c): max(0, int(v)) for c, v in (sell_caps or {}).items()}
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
                bot += min(delta, buy_caps[code]) if code in buy_caps else delta
        elif delta < 0:
            drop = -delta
            if code in sells:
                botpart = min(drop, sell_caps[code]) if code in sell_caps else drop
                manual = int(prev["acct_qty"]) - bot
                rest = drop - botpart                    # [2026-11-25] beyond the held order's size: sold by hand
                bot = max(0, bot - botpart - max(0, rest - manual))   # the bot sold: its part shrinks first
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


# =============================================================================================================================
# [2026-11-25] fill-based ledger (schema 2, bot_fill_ledger.json) - see r023_fill_ledger_plan_20261125.md
# =============================================================================================================================
FILL_LEDGER_FILENAME = "bot_fill_ledger.json"
FILL_SCHEMA = 2
FILL_KEEP_DAYS = 7          # finished registry entries older than this are pruned


def fill_key(ord_dt, odno) -> str:
    """registry key: order date + order number without leading zeros"""
    return f"{str(ord_dt).strip()}:{str(odno).strip().lstrip('0')}"


def new_fill_state(codes: dict[str, int] | None = None) -> dict:
    """codes: {code: bot quantity} - their manual part is unknown (None) until the first account comparison"""
    return {"schema": FILL_SCHEMA,
            "codes": {_norm(c): {"bot_qty": int(q), "manual_qty": None} for c, q in (codes or {}).items() if int(q) > 0},
            "registry": {}, "applied": {}, "deducted": {}}


def load_fill_state(live_runtime_dir: Path) -> tuple[dict | None, str]:
    """(state, "ok") / (None, "missing") / (None, "corrupt")"""
    path = Path(live_runtime_dir) / FILL_LEDGER_FILENAME
    if not path.exists():
        return None, "missing"
    raw = _read_json(path)
    try:
        if raw is None or raw.get("schema") != FILL_SCHEMA:
            raise ValueError("schema")
        codes, reg, applied = raw.get("codes"), raw.get("registry"), raw.get("applied")
        if not isinstance(codes, dict) or not isinstance(reg, dict) or not isinstance(applied, dict):
            raise ValueError("parts")
        state = {"schema": FILL_SCHEMA, "codes": {}, "registry": {}, "applied": {}}
        for c, row in codes.items():
            q = int((row or {}).get("bot_qty", 0))
            m = (row or {}).get("manual_qty")
            state["codes"][_norm(c)] = {"bot_qty": max(0, q), "manual_qty": None if m is None else max(0, int(m))}
        for k, e in reg.items():
            if not isinstance(e, dict) or e.get("side") not in ("buy", "sell"):
                raise ValueError(f"registry {k}")
            state["registry"][str(k)] = {**e, "code": _norm(e.get("code")), "qty": int(e.get("qty", 0) or 0),
                                         "terminal": bool(e.get("terminal"))}
        for k, v in applied.items():
            state["applied"][str(k)] = int(v)
        if raw.get("migrating"):
            state["migrating"] = True
        anchor = raw.get("day_anchor")
        if anchor:
            if not isinstance(anchor, dict) or not isinstance(anchor.get("codes"), dict):
                raise ValueError("day_anchor")
            state["day_anchor"] = {
                "date": str(anchor.get("date", "")),
                "codes": {_norm(c): {"bot_qty": max(0, int((r or {}).get("bot_qty", 0))),
                                     "manual_qty": None if (r or {}).get("manual_qty") is None else max(0, int(r["manual_qty"]))}
                          for c, r in anchor["codes"].items()},
                "applied": {str(k): int(v) for k, v in (anchor.get("applied") or {}).items()},
                "debt": {_norm(c): int(q) for c, q in (anchor.get("debt") or {}).items() if int(q) > 0}}
        for name in ("listed_through", "listed_prev"):
            lt = raw.get(name)
            if lt:
                datetime.strptime(str(lt), "%Y%m%d")
                state[name] = str(lt)
        debt = raw.get("debt") or {}
        if not isinstance(debt, dict):
            raise ValueError("debt")
        state["debt"] = {_norm(c): int(q) for c, q in debt.items() if int(q) > 0}
        pend = raw.get("pending_up") or {}
        if not isinstance(pend, dict):
            raise ValueError("pending_up")
        state["pending_up"] = {}
        for c, v in pend.items():
            q, d = (int(v.get("qty", 0)), str(v.get("date", ""))) if isinstance(v, dict) else (int(v), "")
            if q > 0:
                state["pending_up"][_norm(c)] = {"qty": q, "date": d}
        ded = raw.get("deducted") or {}
        if not isinstance(ded, dict):
            raise ValueError("deducted")
        state["deducted"] = {str(k): v for k, v in ded.items()}
        return state, "ok"
    except (ValueError, TypeError, AttributeError):
        return None, "corrupt"


def save_fill_state(live_runtime_dir: Path, state: dict, now: datetime | None = None) -> None:
    _atomic_write_json(Path(live_runtime_dir) / FILL_LEDGER_FILENAME,
                       {**state, "schema": FILL_SCHEMA, "updated_at": (now or datetime.now()).isoformat(timespec="seconds")})


def migrate_from_ledger(old: dict[str, dict] | None) -> dict:
    """first switch to the fill ledger: the bot quantities of the old ledger are the starting values (user decision 2026-11-25)"""
    return new_fill_state({c: int(r.get("bot_qty", 0) or 0) for c, r in (old or {}).items()})


def register_order(state: dict, ord_dt: str, odno: str, code: str, side: str, qty: int, at: str = "",
                   baseline: bool = False) -> bool:
    """a bot order the broker accepted. False when the order number is empty or it is registered already. baseline: its fills up
    to the next complete listing are not applied (an order recovered while switching from the old ledger)"""
    if not str(odno or "").strip().lstrip("0"):
        return False
    key = fill_key(ord_dt, odno)
    if key in state["registry"]:
        return False
    state["registry"][key] = {"code": _norm(code), "side": "sell" if side == "sell" else "buy", "qty": int(qty), "at": at,
                              "terminal": False, "last_change": "", **({"baseline": True} if baseline else {})}
    return True


def baseline_codes(state: dict | None) -> set[str]:
    """symbols with an order whose baseline is not set yet (their bot part is not known for sure)"""
    return {e["code"] for e in ((state or {}).get("registry") or {}).values() if e.get("baseline")}


def _row_int(row: dict, key: str) -> int:
    try:
        return int(float(str(row.get(key, 0) or 0).strip() or 0))
    except (TypeError, ValueError):
        return 0


def apply_fill_rows(state: dict, rows: list[dict], now: datetime | None = None) -> list[tuple[str, str, int, str]]:
    """Apply the fills of a COMPLETE daily order/fill listing (KIS inquire_daily_ccld output1 rows) to the bot's quantities.
    Only rows of registered bot orders count; a modified order (orgn_odno = a registered order) is registered as its child.
    A row whose symbol / side does not match the registry is not applied (reported with delta 0 and a ':mismatch' key).
    Returns (code, side, delta, key)."""
    now_s = (now or datetime.now()).isoformat(timespec="seconds")
    reg = state["registry"]
    by_key: dict[str, dict] = {}
    for r in rows or []:
        by_key[fill_key(r.get("ord_dt", ""), r.get("odno", ""))] = r
    # lineage: modified orders carry their fills on the new order number
    added = True
    while added:
        added = False
        for key, r in by_key.items():
            org = str(r.get("orgn_odno") or "").strip().lstrip("0")
            if key in reg or not org:
                continue
            parent = reg.get(fill_key(r.get("ord_dt", ""), org))
            if parent is not None:
                reg[key] = {"code": parent["code"], "side": parent["side"], "qty": _row_int(r, "ord_qty"), "at": now_s,
                            "terminal": False, "last_change": "", "child_of": fill_key(r.get("ord_dt", ""), org),
                            **({"baseline": True} if parent.get("baseline") else {})}
                added = True
    out = []
    dropped: list[str] = []
    moves: list[tuple[bool, str, str, int, str, int]] = []   # (is_sell, who, code, delta, key, tot) - applied below, sells first
    for key, e in reg.items():
        r = by_key.get(key)
        if e.get("baseline"):
            # [2026-11-25] an order recovered when switching from the old ledger: its fills so far are already in the inherited bot
            # quantity (or were deliberately not counted) - only fills after the first listing that SHOWS it are applied
            if r is None:
                if key.split(":", 1)[0] < (now or datetime.now()).strftime("%Y%m%d"):
                    dropped.append(key)               # a past day's order the broker does not list: not an order
                continue
            if _norm(r.get("pdno", "")) == e["code"]:
                state["applied"][key] = _row_int(r, "tot_ccld_qty")
                e["base"] = state["applied"][key]     # for the day bound: these fills are in the inherited quantity
                e.pop("baseline", None)
                out.append((e["code"], e["side"], 0, key + ":baseline"))
            continue
        if r is None:
            continue
        side_cd = str(r.get("sll_buy_dvsn_cd", "")).strip()
        row_side = "sell" if side_cd == "01" else "buy" if side_cd == "02" else ""
        if _norm(r.get("pdno", "")) != e["code"] or (row_side and row_side != e["side"]):
            out.append((e["code"], e["side"], 0, key + ":mismatch"))
            continue
        tot = _row_int(r, "tot_ccld_qty")
        delta = tot - int(state["applied"].get(key, 0))
        if delta > 0:
            moves.append((e["side"] == "sell", "bot", e["code"], delta, key, tot))
            e["last_change"] = now_s
        ord_qty = _row_int(r, "ord_qty")
        if (_row_int(r, "rmn_qty") <= 0 or str(r.get("cncl_yn", "")).strip().upper() == "Y"
                or (ord_qty > 0 and tot + _row_int(r, "cncl_cfrm_qty") + _row_int(r, "rjct_qty") >= ord_qty)):
            if not e.get("terminal"):
                e["terminal"] = True
                e["last_change"] = e.get("last_change") or now_s
    for key in dropped:
        reg.pop(key, None)
        out.append(("", "", 0, key + ":dropped"))
    # [v3] orders that are not the bot's (MTS / HTS by hand): their fills change the manual part; a hand sale takes the manual part
    # first (user decision 2026-11-25), the rest comes off the bot part
    for key, r in by_key.items():
        if key in reg:
            continue
        code = _norm(r.get("pdno", ""))
        side_cd = str(r.get("sll_buy_dvsn_cd", "")).strip()
        if side_cd not in ("01", "02") or not code.strip("0"):
            continue
        tot = _row_int(r, "tot_ccld_qty")
        delta = tot - int(state["applied"].get(key, 0))
        if delta > 0:
            moves.append((side_cd == "01", "manual", code, delta, key, tot))
    # [Codex 7th] the listing has no fill times: all sells first, then the buys - the order that leaves the bot part smallest.
    # [Codex 8th] what a sell could not take (both parts empty at that point) was sold out of a buy of this same listing: it is
    # paid from the buys that follow - the bot's buys first (the bot part stays smallest)
    # [Codex 17th] kept between listings (state["debt"]): the buy that pays may show up in a later listing than the sale
    debt: dict[str, int] = state.setdefault("debt", {})
    ordered = sorted(moves, key=lambda m: (not m[0], m[1] != "bot"))   # sells, then bot buys, then buys by hand
    for is_sell, who, code, delta, key, tot in ordered:
        row = _code_row(state, code)
        if is_sell:
            # [Codex 11th] sells take only the CONFIRMED manual part: what the account comparison added ahead of its fill
            # (pending_up) is not there to pay for a sale that came before that buy
            pend_q = int(((state.get("pending_up") or {}).get(code) or {}).get("qty", 0))
            confirmed = max(0, (row["manual_qty"] or 0) - pend_q)
            if who == "bot":                        # the bot's sell: its own part first, the rest from the manual part
                take_b = min(row["bot_qty"], delta)
                row["bot_qty"] -= take_b
                left = delta - take_b
                take_m = min(confirmed, left)
                if row["manual_qty"] is not None:
                    row["manual_qty"] -= take_m
            else:                                   # a hand sale: the manual part first, the rest from the bot part
                take_m = min(confirmed, delta)
                if row["manual_qty"] is not None:
                    row["manual_qty"] -= take_m
                left = delta - take_m
                take_b = min(row["bot_qty"], left)
                row["bot_qty"] -= take_b
            unpaid = delta - take_b - take_m
            if unpaid > 0:
                debt[code] = debt.get(code, 0) + unpaid
            out.append((code, ("sell" if who == "bot" else "manual_sell"), delta, key))
        else:
            pay = min(debt.get(code, 0), delta)
            debt[code] = debt.get(code, 0) - pay
            if who == "bot":
                row["bot_qty"] += delta - pay
                out.append((code, "buy", delta, key))
            else:
                # [Codex 9th] the account showed this buy before the listing did: the comparison already put it in the manual part
                pend = state.setdefault("pending_up", {})
                seen = min(int((pend.get(code) or {}).get("qty", 0)), delta - pay)
                if seen:
                    pend[code]["qty"] -= seen
                    if pend[code]["qty"] <= 0:
                        pend.pop(code, None)
                if row["manual_qty"] is not None:
                    row["manual_qty"] += delta - pay - seen
                out.append((code, "manual_buy", delta, key))
        state["applied"][key] = tot
    for code in [c for c, q in debt.items() if q <= 0]:
        debt.pop(code, None)
    return out


def _code_row(state: dict, code: str) -> dict:
    return state["codes"].setdefault(_norm(code), {"bot_qty": 0, "manual_qty": None})


def busy_codes(rows: list[dict]) -> set[str]:
    """[v3] symbols with any order still open in the listing (bot or by hand) - not compared with the account yet"""
    out = set()
    for r in rows or []:
        if _row_int(r, "rmn_qty") > 0 and str(r.get("cncl_yn", "")).strip().upper() != "Y":
            out.add(_norm(r.get("pdno", "")))
    return out


def open_order_dates(state: dict) -> list[str]:
    """order dates of registered orders that are not finished (to be queried again)"""
    return sorted({k.split(":", 1)[0] for k, e in state["registry"].items() if not e.get("terminal")})


def deduct_once(state: dict, key: str, code: str, qty: int) -> tuple[int, int] | None:
    """[2026-11-25] take qty off the bot part for a bot sell whose fills cannot be seen (no order number) - once per key, also
    across restarts (state["deducted"]). None when this key was taken off already. Returns (old, new)."""
    ded = state.setdefault("deducted", {})
    if key in ded:
        return None
    code = _norm(code)
    old = fill_bot_quantity(state, code)
    new = max(0, old - int(qty))
    _code_row(state, code)["bot_qty"] = new
    ded[key] = {"code": code, "qty": int(qty), "from": old}
    return old, new


def reanchor_code(state: dict, code: str, rows: list[dict] | None, today: str) -> None:
    """[Codex 14th] the symbol's manual part was just set: from here on the day bound starts at these quantities and at the
    counts applied so far of its orders of today (rows: the complete listing of this round)"""
    anchor = state.get("day_anchor") or {}
    if anchor.get("date") != today or rows is None:
        return
    code = _norm(code)
    row = _code_row(state, code)
    anchor.setdefault("codes", {})[code] = {"bot_qty": int(row["bot_qty"]), "manual_qty": row["manual_qty"]}
    anchor.setdefault("debt", {}).pop(code, None)
    applied = anchor.setdefault("applied", {})
    for r in rows:
        if str(r.get("ord_dt", "")).strip() == today and _norm(r.get("pdno", "")) == code:
            key = fill_key(today, r.get("odno", ""))
            applied[key] = int(state["applied"].get(key, 0))


def reconcile_account(state: dict, account_qty: dict[str, int], stable: set[str], today: str | None = None,
                      now: datetime | None = None, confirm_seconds: int = 0,
                      rows: list[dict] | None = None) -> list[tuple[str, int, int]]:
    """[v3] Compare with the account. A symbol whose manual part is unknown starts with account - bot part (the bot part is cut to the
    account if larger). A STABLE symbol (no open order, no fill change in this listing, listing newer than the account read - the
    caller decides) takes the difference into the manual part: more = manual; less = off the manual part first, the rest off the bot
    part. The bot part is never raised here. Returns (code, old_bot, new_bot) of the bot changes."""
    out = []
    pend = state.setdefault("pending_up", {})
    today = today or datetime.now().strftime("%Y%m%d")
    now = now or datetime.now()
    waiting = state.setdefault("init_seen", {})   # [Codex 13th] {code: {qty, at}} - an unknown manual part waits for a second look

    def _pend(code: str, qty: int) -> None:     # [Codex 9th/10th] kept apart from the manual part, until its row is seen
        e = pend.setdefault(code, {"qty": 0, "date": today})
        e["qty"] += qty
        e["date"] = today
    acct = {_norm(c): int(q) for c, q in (account_qty or {}).items() if int(q) > 0}
    for code in sorted(set(state["codes"]) | set(acct)):
        row = _code_row(state, code)
        a = acct.get(code, 0)
        old = row["bot_qty"]
        if code in stable:
            (state.get("debt") or {}).pop(code, None)   # [Codex 17th] matched with the account now: nothing left to pay
        if code not in stable:
            waiting.pop(code, None)                 # moving (or the listing is older than the account): nothing now
        elif row["manual_qty"] is None:
            row["bot_qty"] = min(old, a)
            # [Codex 13th] the account may show a buy by hand the listing does not show yet: the first value only after the same
            # account quantity was stable twice, confirm_seconds apart (its row is listed - and counted as applied - in between)
            seen = waiting.get(code)
            if confirm_seconds <= 0 or (seen and seen.get("qty") == a and
                                        (now - datetime.fromisoformat(seen["at"])).total_seconds() >= confirm_seconds):
                row["manual_qty"] = max(0, a - row["bot_qty"])
                waiting.pop(code, None)
                reanchor_code(state, code, rows, today)
            elif not seen or seen.get("qty") != a:
                waiting[code] = {"qty": a, "at": now.isoformat(timespec="seconds")}
        else:
            diff = a - (row["bot_qty"] + row["manual_qty"])
            if diff > 0:
                row["manual_qty"] += diff
                _pend(code, diff)                       # [Codex 9th] taken off again when its fill shows up in the listing
            elif diff < 0:
                take = min(row["manual_qty"], -diff)
                row["manual_qty"] -= take
                row["bot_qty"] = max(0, row["bot_qty"] - (-diff - take))
        if row["bot_qty"] != old:
            out.append((code, old, row["bot_qty"]))
        if a == 0 and row["bot_qty"] == 0 and (row["manual_qty"] or 0) == 0:
            state["codes"].pop(code, None)
            waiting.pop(code, None)
    return out


def set_day_anchor(state: dict, today: str, account_codes: set[str] | None = None) -> None:
    """[Codex 12th] the ledger right before the day's first listing: the start for clamp_to_day_bound. A symbol the account holds
    that the ledger does not know yet gets an unknown manual part (no bound for it today)."""
    codes = {c: {"bot_qty": int(r["bot_qty"]), "manual_qty": r["manual_qty"]} for c, r in state["codes"].items()}
    for c in account_codes or set():
        codes.setdefault(_norm(c), {"bot_qty": 0, "manual_qty": None})
    state["day_anchor"] = {"date": today, "codes": codes,
                           "applied": {k: int(v) for k, v in state["applied"].items() if k.startswith(today + ":")},
                           "debt": {c: int(q) for c, q in (state.get("debt") or {}).items() if int(q) > 0}}


def clamp_to_day_bound(state: dict, rows: list[dict], today: str) -> list[tuple[str, int, int]]:
    """[Codex 12th] The listing has no fill times and fills may show up split over several listings in any order. Bound that does
    not depend on that order: all of today's fills (cumulative counts of this complete listing) applied to the day's anchor with
    every sell first (hand sales: manual part, then bot part; unpaid sells paid by the bot's buys first) = the smallest bot part any
    order of events gives. A larger bot part in the ledger is cut to it, the difference goes to the manual part.
    Not for symbols whose anchor manual part is unknown or that still wait for a baseline. Returns (code, old_bot, new_bot)."""
    anchor = state.get("day_anchor") or {}
    if anchor.get("date") != today:
        return []
    reg, a_app = state["registry"], anchor.get("applied") or {}
    per: dict[str, list[tuple[bool, str, int]]] = {}
    for r in rows or []:
        if str(r.get("ord_dt", "")).strip() != today:
            continue
        code = _norm(r.get("pdno", ""))
        side_cd = str(r.get("sll_buy_dvsn_cd", "")).strip()
        if side_cd not in ("01", "02") or not code.strip("0"):
            continue
        key = fill_key(today, r.get("odno", ""))
        e = reg.get(key)
        who = "manual"
        seen = int(a_app.get(key, 0))
        if e is not None:
            if e["code"] != code or e["side"] != ("sell" if side_cd == "01" else "buy"):
                continue
            seen = max(seen, int(e.get("base", 0) or 0))   # [Codex 13th] both are counts of the same fills: once
            who = "bot"
        qty = _row_int(r, "tot_ccld_qty") - seen
        if qty > 0:
            per.setdefault(code, []).append((side_cd == "01", who, qty))
    out = []
    waiting = baseline_codes(state)
    for code, moves in per.items():
        base = (anchor.get("codes") or {}).get(code) or {"bot_qty": 0, "manual_qty": 0}
        if code in waiting or base.get("manual_qty") is None:
            continue
        b, m, debt = int(base["bot_qty"]), int(base["manual_qty"]), int((anchor.get("debt") or {}).get(code, 0))
        for is_sell, who, q in sorted(moves, key=lambda x: (not x[0], x[1] != "bot")):
            if is_sell:
                first = min(b if who == "bot" else m, q)
                second = min(m if who == "bot" else b, q - first)
                tb, tm = (first, second) if who == "bot" else (second, first)
                b, m, debt = b - tb, m - tm, debt + q - tb - tm
            else:
                pay = min(debt, q)
                debt -= pay
                if who == "bot":
                    b += q - pay
                else:
                    m += q - pay
        row = _code_row(state, code)
        old = row["bot_qty"]
        if b < old:
            row["bot_qty"] = b
            if row["manual_qty"] is not None:
                row["manual_qty"] += old - b
            out.append((code, old, b))
    return out


def day_fill_totals(rows: list[dict] | None, today: str) -> dict[str, tuple[int, int]]:
    """[Codex 15th] {code: (bought, sold)} - the fills of today's orders in a complete listing (bot and by hand)"""
    out: dict[str, list[int]] = {}
    for r in rows or []:
        if str(r.get("ord_dt", "")).strip() != today:
            continue
        side_cd = str(r.get("sll_buy_dvsn_cd", "")).strip()
        if side_cd not in ("01", "02"):
            continue
        out.setdefault(_norm(r.get("pdno", "")), [0, 0])[0 if side_cd == "02" else 1] += _row_int(r, "tot_ccld_qty")
    return {c: (v[0], v[1]) for c, v in out.items()}


def expire_pending(state: dict, start: str) -> None:
    """[Codex 10th] a pending_up entry whose day is before the listing range: its row can no longer come as new - drop it"""
    pend = state.get("pending_up") or {}
    for code in [c for c, e in pend.items() if str(e.get("date", "")) < start]:
        pend.pop(code, None)


def fill_bot_quantity(state: dict | None, code) -> int:
    row = ((state or {}).get("codes") or {}).get(_norm(code))
    return int(row.get("bot_qty", 0)) if isinstance(row, dict) else 0


def reserved_sell(state: dict | None, code) -> int:
    """the part of unfinished bot sell orders of the symbol whose fills are not in the ledger yet (may still be sold)"""
    code = _norm(code)
    total = 0
    for key, e in ((state or {}).get("registry") or {}).items():
        if e.get("code") == code and e.get("side") == "sell" and not e.get("terminal"):
            total += max(0, int(e.get("qty", 0)) - int(state["applied"].get(key, 0)))
    return total


def prune_fill_state(state: dict, today: str, keep_days: int = FILL_KEEP_DAYS) -> None:
    """drop finished registry entries (and their applied counts) older than keep_days"""
    try:
        cutoff = datetime.strptime(today, "%Y%m%d").toordinal() - keep_days
        dates = open_order_dates(state) + [state[n] for n in ("listed_through", "listed_prev") if state.get(n)]
        if dates:   # [Codex 8th/9th] dates still listed again (oldest unfinished order, last listed day) keep their applied counts
            cutoff = min(cutoff, datetime.strptime(min(dates), "%Y%m%d").toordinal())
    except ValueError:
        return
    for key in list(state["registry"]):
        day = key.split(":", 1)[0]
        try:
            old = datetime.strptime(day, "%Y%m%d").toordinal() < cutoff
        except ValueError:
            old = False
        if old and state["registry"][key].get("terminal"):
            state["registry"].pop(key, None)
            state["applied"].pop(key, None)
    for key in list(state["applied"]):
        if key in state["registry"]:
            continue
        try:
            if datetime.strptime(key.split(":", 1)[0], "%Y%m%d").toordinal() < cutoff:
                state["applied"].pop(key, None)     # a manual order's count, older than keep_days
        except ValueError:
            pass
    for key in list(state.get("deducted") or {}):
        try:
            if datetime.strptime(key.split(":", 1)[0], "%Y%m%d").toordinal() < cutoff:
                state["deducted"].pop(key, None)
        except ValueError:
            pass
