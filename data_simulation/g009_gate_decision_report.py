"""
g009_gate_decision_report.py

게이트 판정 기록(jsonl)을 읽어 판정 이후 실제 주가 흐름(사후 수익률)을 집계한다.
추격차단(_019, ANTI_CHASE)과 점심/진입창 관찰(MIDDAY_SHADOW / ENTRY_WINDOW_SHADOW)이 막은 진입이
실제로 나쁜 진입이었는지를 표본으로 쌓기 위한 리포트 (Codex 2차 검토 2026-10-05 권고).

입력:
- live: data/live_runtime/gate_decisions_YYYYMMDD.jsonl   (r003 record_gate_decision)
- sim : <data-root>/YYYYMMDD/YYYYMMDD_gate_decisions[_minute].jsonl   (g003)
가격: data/YYYYMMDD/<code>_*_10s.txt 의 :00 행(실제 1분봉, 분 시작 라벨 -> +1분 = 확정 시각)만 사용한다
      (나머지 10초 행은 보간값이라 사용하지 않음).

사후 수익률: 판정 가격(기록의 price) 대비 판정 후 N분 시점까지 확정된 마지막 실제 1분봉 종가.
EOD = 15:19까지 확정된 마지막 종가(정규장 강제청산 기준). 비용 0.23%는 net 열에만 반영.
같은 종목/종류/결과는 --episode-min 분 안의 첫 판정만 1건으로 센다(폴링 반복 기록 중복 제거).

Update log format (append only):
- [YYYY-MM-DD] type=<feat|fix|refactor|perf|docs|chore|test> owner=<name>
    summary: <what changed and why>
    impact: <live|sim|scanner|common|docs>
    compatibility: <backward-compatible|breaking>

Update log:
- [2026-10-10] type=fix owner=claude
    summary: load_minute_closes가 g001 10s 파일의 raw_bar 열이 있으면 원본 분봉 행만 사용(무체결 분 보간 :00 행 제외).
    impact: sim
    compatibility: backward-compatible (열 없으면 기존 동작)
- [2026-10-05] type=feat owner=claude
    summary: 신규 - 게이트 판정 기록 사후 수익률 리포트(r003/g003 gate_decisions jsonl).
    impact: sim
    compatibility: backward-compatible
"""

from __future__ import annotations

import argparse
import glob
import json
from datetime import time as dt_time
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
COST_PCT = 0.23
EOD_TIME = dt_time(15, 19)


def load_minute_closes(data_dir: Path, date_str: str, code: str) -> pd.Series | None:
    """실제 1분봉 종가 시계열(index = 확정 시각 = 분 시작 + 1분)."""
    files = glob.glob(str(data_dir / date_str / f"{code}_*_10s.txt"))
    if not files:
        return None
    df = pd.read_csv(files[0], encoding="utf-8-sig", usecols=lambda c: c in ("datetime", "close", "raw_bar"))
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df = df[df["datetime"].dt.second == 0]
    if "raw_bar" in df.columns:  # g001 원본 분봉 표식 - 무체결 분의 보간 :00 행 제외
        df = df[pd.to_numeric(df["raw_bar"], errors="coerce") == 1]
    if df.empty:
        return None
    s = pd.Series(pd.to_numeric(df["close"], errors="coerce").to_numpy(),
                  index=df["datetime"] + pd.Timedelta(minutes=1)).dropna()
    return s[~s.index.duplicated(keep="last")].sort_index()


def close_at(series: pd.Series, ts: pd.Timestamp) -> float | None:
    pos = series.index.searchsorted(ts, side="right")
    return None if pos <= 0 else float(series.iloc[pos - 1])


def load_records(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        if not p.exists():
            print(f"[WARN] no file: {p}")
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                rec["_file"] = p.name
                rows.append(rec)
    return pd.DataFrame(rows)


def dedup_episodes(df: pd.DataFrame, episode_min: int) -> pd.DataFrame:
    df = df.sort_values("ts")
    keep, last = [], {}
    for i, r in df.iterrows():
        key = (r["date"], r["code"], r["kind"], r["result"])
        if key in last and (r["ts"] - last[key]).total_seconds() < episode_min * 60:
            continue
        last[key] = r["ts"]
        keep.append(i)
    return df.loc[keep]


def main() -> None:
    ap = argparse.ArgumentParser(description="게이트 판정 기록 사후 수익률 리포트")
    ap.add_argument("--dates", nargs="+", required=True, help="YYYYMMDD ...")
    ap.add_argument("--source", choices=("live", "sim"), default="live")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="가격 데이터 폴더(data/YYYYMMDD)")
    ap.add_argument("--sim-root", default=None, help="sim: g003 --data-root (기본 = --data-dir)")
    ap.add_argument("--sim-suffix", default="", help="sim: 파일 접미사 (예: _minute)")
    ap.add_argument("--horizons", nargs="+", type=int, default=[10, 30, 60])
    ap.add_argument("--episode-min", type=int, default=10)
    ap.add_argument("--csv", default=None, help="종목별 상세 CSV 저장 경로")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if args.source == "live":
        paths = [data_dir / "live_runtime" / f"gate_decisions_{d}.jsonl" for d in args.dates]
    else:
        root = Path(args.sim_root or args.data_dir)
        paths = [root / d / f"{d}_gate_decisions{args.sim_suffix}.jsonl" for d in args.dates]
    df = load_records(paths)
    if df.empty:
        print("no records")
        return
    df["ts"] = pd.to_datetime(df["ts"])
    df["date"] = df["ts"].dt.strftime("%Y%m%d")
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = dedup_episodes(df, args.episode_min)

    cache: dict[tuple[str, str], pd.Series | None] = {}
    out = []
    for _, r in df.iterrows():
        key = (r["date"], r["code"])
        if key not in cache:
            cache[key] = load_minute_closes(data_dir, *key)
        s = cache[key]
        price = float(r.get("price") or 0)
        row = r.to_dict()
        for h in args.horizons:
            px = close_at(s, r["ts"] + pd.Timedelta(minutes=h)) if (s is not None and price > 0) else None
            row[f"fwd{h}"] = None if px is None else (px / price - 1.0) * 100.0
        eod_ts = pd.Timestamp.combine(r["ts"].date(), EOD_TIME)
        px = close_at(s, eod_ts) if (s is not None and price > 0 and r["ts"] < eod_ts) else None
        row["fwd_eod"] = None if px is None else (px / price - 1.0) * 100.0
        out.append(row)
    res = pd.DataFrame(out)
    if "would_block" not in res.columns:
        res["would_block"] = None

    fcols = [f"fwd{h}" for h in args.horizons] + ["fwd_eod"]
    grp = res.groupby(["kind", "result", "would_block"], dropna=False)
    table = grp[fcols].mean().round(3)
    table.insert(0, "n", grp.size())
    for c in fcols:
        table[f"win_{c}"] = grp[c].apply(lambda x: (x.dropna() > COST_PCT).mean()).round(3)
    table["net_eod"] = (table["fwd_eod"] - COST_PCT).round(3)
    pd.set_option("display.width", 200)
    print(f"source={args.source} dates={args.dates} episodes={len(res)} (episode window {args.episode_min}m, cost {COST_PCT}%)")
    print("fwdN = 판정가 대비 N분 후 실제 1분봉 종가 수익률(%), win_* = 비용(0.23%) 초과 비율")
    print(table.to_string())
    if args.csv:
        res.to_csv(args.csv, index=False, encoding="utf-8-sig")
        print(f"saved {args.csv}")


if __name__ == "__main__":
    main()
