# -*- coding: utf-8 -*-
"""g061_s1_shadow_log.py - 실거래에 쓰지 않는 쪽 프로파일(s1 / v0)의 선정을 매일 따로 기록한다 (병행 기록, 주문 / 감시목록 영향 없음).

[2026-11-28 저녁] 실거래 기본값이 s1 이 되었다: 이제 선별 파일 = s1, 이 스크립트가 따로 돌리는 쪽 = v0 (아래 설명의 반대).
기록 형식은 같다 - picks 는 언제나 s1 목록, v0_picks 는 언제나 v0 목록이고 live_profile 이 어느 쪽이 실거래였는지 말한다.
한계: v0 의 반복 선정 감점은 이전 날짜의 선별 파일을 읽으므로, 실거래가 s1 인 동안의 v0 기록은 "v0 를 계속 썼을 때"와 조금 다를 수 있다.

Why: 2026-11-28 재현(6/30 ~ 11/27)에서 s1 목록이 현재 선정(v0)보다 조금 나았지만 s1 은 그 기간 자료에서 고른 규칙이다.
실거래 선정을 바꾸기 전에 새 기간에서 같은 비교를 다시 하려면, 그날그날 s1 이 무엇을 골랐는지가 남아 있어야 한다
(나중에 다시 계산하면 그 사이 고친 코드 / 보정된 자료가 섞인다).

What: 날짜 D 에 대해 g002.scan 을 G002_FILTER_PROFILE=s1 로 이 프로세스 안에서만 한 번 더 돌린다 (g043 재현과 같은 호출).
  - 선별 파일(_D_picks.txt), r004 감시목록, 리포트는 건드리지 않는다. KIS 호출 없음.
  - 같은 시각의 v0 선별 파일 내용도 함께 적는다 (v0 를 나중에 다시 돌려도 그날의 비교 대상이 남도록).
Files (<runtime>/data/pick_shadow_s1/):
  picks_D.json   {date, for_trading_day, profile, logged_at, eligible, picks: [{rank, code, name, score, price, amount_ma20,
                  vol_ratio, tick_ratio}], v0_picks: [{code, name}] | null}
  이미 있는 날짜는 다시 쓰지 않는다 (--force 로만 덮어씀) - 전진 기록은 처음 적은 값이 기준이다.
Run:
  python g061_s1_shadow_log.py --date 20261130          하루 기록 (r021 postclose 단계 "s1log")
  python g061_s1_shadow_log.py --watch-json OUT.json    기록된 날짜 전부를 g058 --watch-json 형식 두 개로 묶는다
                                                        (OUT.json = s1, OUT 이름에 _v0 를 붙인 파일 = v0; 키 = 다음 거래일)
Exit codes: 0 기록함 / 이미 있음 / 묶음 완료, 2 그 날짜의 자료가 없거나 스캔 실패.

Update log:
- [2026-11-28] type=feat owner=claude
    summary: 실거래 기본 프로파일이 s1 이 됨 - 실거래가 s1 이면 v0 를 따로 돌려 기록 (live_profile / shadow_profile 필드 추가).
    impact: research tooling. 기록 형식 호환 (picks = s1, v0_picks = v0 그대로)
    compatibility: backward-compatible
- [2026-11-28] type=feat owner=claude
    summary: 신규 - s1 병행 기록 (사용자 요청: s1 을 바로 전환하지 않고 매일 따로 기록해 새 기간으로 다시 비교).
    impact: research tooling (runtime/data/pick_shadow_s1). 주문 / r004 / 선별 파일 불변
    compatibility: backward-compatible (신규 파일)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "live_trading"))

import g002_data_scan_trade_candidates as g002  # noqa: E402
import g005_watchlist_bridge as g005  # noqa: E402

OUT_DIR = g002.RUNTIME_DATA_DIR / "pick_shadow_s1"


def shadow_path(d: str) -> Path:
    return OUT_DIR / f"picks_{d}.json"


def file_picks(d: str) -> list[dict] | None:
    """the picks file g002 wrote for d ("code,name" lines); None when there is none"""
    p = g002.RUNTIME_DATA_DIR / d / f"_{d}_picks.txt"
    if not p.is_file():
        return None
    out = []
    for line in p.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        code, _, name = line.partition(",")
        out.append({"code": code.strip(), "name": name.strip()})
    return out


def shadow_profile() -> str:
    """the profile that is NOT live: v0 while the live selection is s1, s1 otherwise"""
    return "v0" if g002._filter_profile() == "s1" else "s1"


def scan_profile(d: str, profile: str) -> dict:
    """g002.scan of date d under the given profile, in this process only (the environment variable is put back)."""
    root = g002.RUNTIME_DATA_DIR
    base = g002.filter_stock_list_by_existing_data(g002.load_symbols(), root, verbose=False)
    sl = g002.filter_stock_list_by_min_bars(base, root, target_date_str=d, min_bars=g002.MIN_REQUIRED_BARS, verbose=False)
    before = os.environ.get("G002_FILTER_PROFILE")
    os.environ["G002_FILTER_PROFILE"] = profile   # g002 reads it per call; worker processes inherit it
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res = g002.scan(root, datetime.strptime(d, "%Y%m%d"), config=g002.DEFAULT_CONFIG, return_details=True,
                            verbose=False, stock_list=sl)
    finally:
        if before is None:
            os.environ.pop("G002_FILTER_PROFILE", None)
        else:
            os.environ["G002_FILTER_PROFILE"] = before
    rows = res["all_rows"]
    picked = sorted((r for r in rows if r.get("picked")), key=lambda r: -(r.get("score") or 0.0))
    return {"eligible": sum(1 for r in rows if r.get("eligible")),
            "picks": [{"rank": i, "code": r["code"], "name": r.get("name"), "score": r.get("score"), "price": r.get("price"),
                       "amount_ma20": r.get("amount_ma20"), "vol_ratio": r.get("s1_vol_ratio"), "tick_ratio": r.get("s1_tick_ratio")}
                      for i, r in enumerate(picked, start=1)]}


def log_date(d: str, force: bool = False) -> int:
    out = shadow_path(d)
    if out.is_file() and not force:
        print(f"[g061] {d}: already logged ({out}) - kept (--force to overwrite)")
        return 0
    if not (g002.RUNTIME_DATA_DIR / d).is_dir():
        print(f"[g061] {d}: no data folder")
        return 2
    live, shadow = g002._filter_profile(), shadow_profile()
    try:
        res = scan_profile(d, shadow)
    except (Exception, SystemExit) as e:  # g002 exits when the date has no data
        print(f"[g061] {d}: scan failed - {e!r}")
        return 2
    from_file = file_picks(d)   # the live selection of d (the picks file g002 wrote)
    s1, v0 = (from_file, res["picks"]) if live == "s1" else (res["picks"], from_file)
    if live == "s1":
        s1 = None if s1 is None else [{"rank": i, **p} for i, p in enumerate(s1, start=1)]
        v0 = [{"code": p["code"], "name": p["name"]} for p in v0]
    rec = {"date": d, "for_trading_day": g005.next_trading_day(d), "profile": "s1", "live_profile": live, "shadow_profile": shadow,
           "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "eligible": res["eligible"], "picks": s1, "v0_picks": v0}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(out)
    both = len({p["code"] for p in s1} & {p["code"] for p in v0}) if s1 is not None and v0 is not None else 0
    print(f"[g061] {d} -> {rec['for_trading_day']}: live {live}, s1 {'-' if s1 is None else len(s1)}, "
          f"v0 {'-' if v0 is None else len(v0)}, in both {both} (shadow {shadow}: eligible {rec['eligible']}) | {out}")
    return 0


def write_watch_json(path: Path) -> int:
    """every logged date -> {trading day: [codes]} for g058 --watch-json: <path> (s1) and <stem>_v0<suffix> (v0)"""
    s1, v0 = {}, {}
    for f in sorted(OUT_DIR.glob("picks_*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        day = rec.get("for_trading_day")
        if not day:
            continue
        if rec.get("picks") is not None:
            s1[day] = [p["code"] for p in rec["picks"]]
        if rec.get("v0_picks") is not None:
            v0[day] = [p["code"] for p in rec["v0_picks"]]
    path.write_text(json.dumps(s1), encoding="utf-8")
    p0 = path.with_name(path.stem + "_v0" + path.suffix)
    p0.write_text(json.dumps(v0), encoding="utf-8")
    print(f"[g061] s1 {len(s1)} days -> {path} | v0 {len(v0)} days -> {p0}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYYMMDD")
    ap.add_argument("--force", action="store_true", help="overwrite the record of that date")
    ap.add_argument("--watch-json", help="write the logged picks as g058 --watch-json files")
    a = ap.parse_args()
    if a.watch_json:
        return write_watch_json(Path(a.watch_json))
    if not a.date:
        ap.error("--date or --watch-json")
    return log_date(a.date, a.force)


if __name__ == "__main__":
    os.chdir(HERE)   # g002 reads g004_universe_symbols_master.txt from the working directory
    for _s in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            _s.reconfigure(encoding="utf-8")
    sys.exit(main())
