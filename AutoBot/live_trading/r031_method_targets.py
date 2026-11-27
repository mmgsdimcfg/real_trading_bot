# -*- coding: utf-8 -*-
"""Per-method target symbols, composed from the scanner's output (user request 2026-11-27).

    A          = the g002 picks of a scan date (<runtime>/data/D/_D_picks.txt)
    rule       = A + the next symbols of the scan ranking (_D_scan_all.md, score order) up to SIZES["rule"] (30)
    Edge AI    = A + the symbols after those in the ranking up to SIZES["edge"] (50)
    VWAP       = A + the rule extras + the Edge AI extras
    each list  = (composed list + the user's "always include") - the user's "always exclude"   (kept from day to day)
    watchlist  = the union, in this order: A, rule extras + rule includes, VWAP includes, Edge AI extras + Edge AI includes

Pure functions + the config file <runtime>/config/method_targets.json. Nothing here trades or reads a price: it only says which
symbols go into r004 and into r001 RULE_ / EDGE_ / VWAP_TARGET_CODES (r016 `targets apply` writes them; r003 reads them at its
next start). "Edge AI selects" = the scan ranking for now (user decision 2026-11-27: no model-based symbol choice exists yet).

Things the user was told and the app shows:
  - the symbols after A in the ranking are mostly ones the scanner did NOT pick (fail_reasons / eligible=N) - each row keeps its
    rank, score, eligible flag and fail reasons so the choice can be judged before it is applied;
  - ETF / ETN names are skipped when exclude_funds is on (the side strategies do not buy them anyway);
  - the tick collector subscribes at most TICK_CAP (40) codes of the watchlist: the symbols after that have no tick feed, so the
    rule trade and the VWAP trade cannot buy them (no_tick) - the order above puts A and the rule extras first.

Update log:
- [2026-11-27] type=feat owner=claude
    summary: 신규 - 매매법별 대상 종목 자동 구성(A + 스캔 순위, 규칙 30 / Edge AI 50 / VWAP 합집합) + 방식별 항상 포함 / 항상 제외.
    impact: live (r016 targets apply 를 실행하고 r003 을 재시작했을 때만; 기본 auto=False 라 장 마감 작업은 지금까지와 같음)
    compatibility: backward-compatible (신규 파일, 설정 파일이 없으면 아무것도 하지 않음)
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

METHODS = ("rule", "edge", "vwap")
CONFIG_FILE = "method_targets.json"
DEFAULT_SIZES = {"rule": 30, "edge": 50}
SIZE_RANGE = {"rule": (1, 100), "edge": (1, 100)}
TICK_CAP = 40                    # g011 MAX_SUBSCRIPTIONS / r003 _TICK_COLLECTOR_MAX_CODES
WATCH_MAX = 100                  # r016 WATCHLIST_MAX_CODES
EDIT_MAX = 100                   # codes in one include / exclude list
_CODE = re.compile(r"[0-9A-Z]{6}")


def default_config() -> dict:
    return {"auto": False, "sizes": dict(DEFAULT_SIZES), "exclude_funds": True,
            "include": {m: [] for m in METHODS}, "exclude": {m: [] for m in METHODS}}


def _codes(value) -> list[str]:
    out: list[str] = []
    for x in value if isinstance(value, (list, tuple)) else []:
        c = str(x).strip().upper()
        if _CODE.fullmatch(c) and c not in out:
            out.append(c)
    return out[:EDIT_MAX]


def normalize_config(raw) -> dict:
    """a config with every field present and valid (unknown fields dropped, bad values replaced by the default)"""
    cfg = default_config()
    if not isinstance(raw, dict):
        return cfg
    cfg["auto"] = raw.get("auto") is True
    cfg["exclude_funds"] = raw.get("exclude_funds") is not False
    sizes = raw.get("sizes") if isinstance(raw.get("sizes"), dict) else {}
    for m, (lo, hi) in SIZE_RANGE.items():
        v = sizes.get(m)
        if isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi:
            cfg["sizes"][m] = v
    for kind in ("include", "exclude"):
        src = raw.get(kind) if isinstance(raw.get(kind), dict) else {}
        for m in METHODS:
            cfg[kind][m] = _codes(src.get(m))
    for m in METHODS:                                # a code cannot be both: the newer intent is not known, exclude wins
        cfg["include"][m] = [c for c in cfg["include"][m] if c not in cfg["exclude"][m]]
    for k in ("updated_at", "updated_by"):
        if isinstance(raw.get(k), str):
            cfg[k] = raw[k]
    return cfg


def config_path(config_dir: Path) -> Path:
    return Path(config_dir) / CONFIG_FILE


def load_config(config_dir: Path) -> tuple[dict, str]:
    """(config, state): state "missing" (defaults), "ok", or "broken" (defaults with auto off - nothing is composed from a file
    that cannot be read)"""
    p = config_path(config_dir)
    if not p.is_file():
        return default_config(), "missing"
    try:
        return normalize_config(json.loads(p.read_text(encoding="utf-8"))), "ok"
    except Exception:
        return default_config(), "broken"


def save_config(config_dir: Path, cfg: dict, by: str = "") -> dict:
    out = normalize_config(cfg)
    out["updated_at"] = datetime.now().isoformat(timespec="seconds")
    out["updated_by"] = by
    p = config_path(config_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)
    return out


def merge_config(cfg: dict, patch) -> tuple[dict, str | None]:
    """the config with a partial change applied ({auto, sizes: {rule, edge}, exclude_funds, include: {method: [...]},
    exclude: {...}}); (config, None) or (unchanged config, why not)"""
    if not isinstance(patch, dict):
        return cfg, "expected a JSON object"
    new = json.loads(json.dumps(normalize_config(cfg)))
    edits = ("include_add", "include_remove", "exclude_add", "exclude_remove")
    for k in patch:
        if k not in ("auto", "sizes", "exclude_funds", "include", "exclude", *edits):
            return cfg, f"unknown field {k!r}"
    # [Codex review] {"exclude_add": {"vwap": ["005930"]}} etc.: applied to the lists stored NOW (inside r016's lock), so an edit
    # made elsewhere in the meantime is not overwritten by a list the client read earlier
    for k in edits:
        if k not in patch:
            continue
        kind, op = k.split("_")
        if not isinstance(patch[k], dict):
            return cfg, f"{k} must be an object"
        for m, v in patch[k].items():
            if m not in METHODS:
                return cfg, f"{k}: unknown method {m!r}"
            if not isinstance(v, list) or any(not isinstance(x, str) or not _CODE.fullmatch(x.strip().upper()) for x in v):
                return cfg, f"{k}.{m} must be a list of 6-character codes"
            if len(v) > EDIT_MAX:                 # [Codex review] refused, not cut silently
                return cfg, f"{k}.{m}: too many codes ({len(v)} > {EDIT_MAX})"
            codes = _codes(v)
            other = "exclude" if kind == "include" else "include"
            if op == "add":
                new[kind][m] = new[kind][m] + [c for c in codes if c not in new[kind][m]]   # a code already there keeps its place
                new[other][m] = [c for c in new[other][m] if c not in codes]
                if len(new[kind][m]) > EDIT_MAX:
                    return cfg, f"{kind}.{m}: too many codes (> {EDIT_MAX})"
            else:
                new[kind][m] = [c for c in new[kind][m] if c not in codes]
    if "auto" in patch:
        if not isinstance(patch["auto"], bool):
            return cfg, "auto must be true / false"
        new["auto"] = patch["auto"]
    if "exclude_funds" in patch:
        if not isinstance(patch["exclude_funds"], bool):
            return cfg, "exclude_funds must be true / false"
        new["exclude_funds"] = patch["exclude_funds"]
    if "sizes" in patch:
        if not isinstance(patch["sizes"], dict):
            return cfg, "sizes must be an object"
        for m, v in patch["sizes"].items():
            if m not in SIZE_RANGE:
                return cfg, f"sizes: unknown method {m!r}"
            lo, hi = SIZE_RANGE[m]
            if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
                return cfg, f"sizes.{m} must be a whole number {lo}..{hi}"
            new["sizes"][m] = v
    sent: set[tuple[str, str]] = set()
    for kind in ("include", "exclude"):
        if kind not in patch:
            continue
        if not isinstance(patch[kind], dict):
            return cfg, f"{kind} must be an object"
        for m, v in patch[kind].items():
            if m not in METHODS:
                return cfg, f"{kind}: unknown method {m!r}"
            if not isinstance(v, list) or any(not isinstance(x, str) or not _CODE.fullmatch(x.strip().upper()) for x in v):
                return cfg, f"{kind}.{m} must be a list of 6-character codes"
            if len(v) > EDIT_MAX:
                return cfg, f"{kind}.{m}: too many codes ({len(v)} > {EDIT_MAX})"
            new[kind][m] = _codes(v)
            sent.add((kind, m))
    # a list sent now is the newer intent: its codes leave the opposite list of that method when that one was not sent too
    for kind, m in sent:
        other = "exclude" if kind == "include" else "include"
        if (other, m) not in sent:
            new[other][m] = [c for c in new[other][m] if c not in new[kind][m]]
    return normalize_config(new), None


def parse_scan_all(path: Path) -> list[dict]:
    """rows of g002 _<date>_scan_all.md in rank order: {rank, code, name, score, picked, eligible, fail}. [] when unreadable."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    head: list[str] | None = None
    rows: list[dict] = []
    seen: set[str] = set()
    for line in text.splitlines():
        cols = [c.strip() for c in line.strip().strip("|").replace("\\|", "/").split("|")]
        if head is None:
            if cols and cols[0] == "rank":
                head = cols
            continue
        if not cols or not cols[0].isdigit():
            continue
        r = dict(zip(head, cols))
        code = str(r.get("code", "")).upper()
        if not _CODE.fullmatch(code) or code in seen:
            continue
        seen.add(code)
        try:
            score = float(r.get("score", ""))
        except ValueError:
            score = None
        rows.append({"rank": int(cols[0]), "code": code, "name": r.get("name") or code, "score": score,
                     "picked": bool(str(r.get("picked", "")).strip()), "eligible": str(r.get("eligible", "")).strip().upper() == "Y",
                     # what the scan held against it: the fail reasons and / or why it was skipped [Codex review]
                     "fail": " / ".join(x for x in (str(r.get("fail_reasons", "")).strip(),
                                                    ("skip:" + str(r.get("skip_reason", "")).strip()) if str(r.get("skip_reason", "")).strip() else "") if x)})
    return rows


def compose(picks: list[dict], ranked: list[dict], cfg: dict, is_fund=None, names: dict | None = None, date: str = "") -> dict:
    """The three target lists and the watchlist for one scan. picks = A ([{code, name}] in scanner order), ranked = parse_scan_all.
    is_fund(name) -> bool (ETF / ETN names; None = no filter). names: {code: name} for manual codes the scan does not know.
    Returns {a, rule_extra, edge_extra, lists: {rule, edge, vwap}, watch: [{code, name}], rows: {code: {...}}, no_tick, short,
    dropped (codes that did not fit WATCH_MAX), sizes}."""
    cfg = normalize_config(cfg)
    info: dict[str, dict] = {r["code"]: r for r in ranked}
    name_of = {**{r["code"]: r["name"] for r in ranked}, **(names or {}), **{p["code"]: p.get("name") or p["code"] for p in picks}}
    a = [p["code"] for p in picks]
    a_set = set(a)
    skip_fund = cfg["exclude_funds"] and is_fund is not None

    def usable(r: dict) -> bool:
        if r["code"] in a_set:
            return False
        return not (skip_fund and is_fund(r.get("name")))

    pool = [r["code"] for r in ranked if usable(r)]
    n_rule = max(0, cfg["sizes"]["rule"] - len(a))
    n_edge = max(0, cfg["sizes"]["edge"] - len(a))
    rule_extra = pool[:n_rule]
    edge_extra = pool[n_rule:n_rule + n_edge]

    auto = {"rule": a + rule_extra, "edge": a + edge_extra, "vwap": a + rule_extra + edge_extra}
    lists: dict[str, list[str]] = {}
    for m in METHODS:
        ex = set(cfg["exclude"][m])
        out = [c for c in auto[m] if c not in ex]
        out += [c for c in cfg["include"][m] if c not in ex and c not in out]
        lists[m] = out

    order: list[str] = []
    for part in (a, rule_extra, cfg["include"]["rule"], cfg["include"]["vwap"], edge_extra, cfg["include"]["edge"]):
        for c in part:
            if c not in order and any(c in lists[m] for m in METHODS):
                order.append(c)
    dropped = order[WATCH_MAX:]
    order = order[:WATCH_MAX]
    if dropped:
        gone = set(dropped)
        lists = {m: [c for c in v if c not in gone] for m, v in lists.items()}

    rows: dict[str, dict] = {}
    for c in order:
        r = info.get(c) or {}
        src = "A" if c in a_set else "rule" if c in rule_extra else "edge" if c in edge_extra else "manual"
        rows[c] = {"name": name_of.get(c, c), "src": src, "rank": r.get("rank"), "score": r.get("score"),
                   "eligible": r.get("eligible"), "fail": r.get("fail", ""), "in": [m for m in METHODS if c in lists[m]],
                   "included": [m for m in METHODS if c in cfg["include"][m]]}
    return {
        "a": a, "rule_extra": rule_extra, "edge_extra": edge_extra, "lists": lists,
        "watch": [{"code": c, "name": name_of.get(c, c)} for c in order],
        "rows": rows, "tick_cap": TICK_CAP, "no_tick": order[TICK_CAP:], "dropped": dropped,
        "sizes": dict(cfg["sizes"]),
        # fewer extras than asked for: the ranking ran out (a short / missing scan_all file)
        "short": {"rule": n_rule - len(rule_extra), "edge": n_edge - len(edge_extra)},
        "excluded": {m: [c for c in cfg["exclude"][m] if c in auto[m]] for m in METHODS},
        # what `apply` stores: the scan date (r004 header), each list as a set, the watch order [Codex review]
        "digest": digest({"date": date, "lists": {m: sorted(v) for m, v in lists.items()}, "watch": order}),
    }


def digest(obj) -> str:
    """a short fingerprint of a composition: `targets apply --expect` refuses when what would be stored is not what was shown"""
    import hashlib
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=True).encode("ascii")).hexdigest()[:16]

