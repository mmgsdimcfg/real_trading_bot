import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ScannerConfig:
    name: str
    price_min: int
    price_max: int
    atr_ratio_min: float
    volume_ma20_min: int
    amount_ma20_min: int
    require_trend: bool
    max_picks: int | None


CURRENT_STRICT_CONFIG = ScannerConfig(
    name="current_strict",
    price_min=5000,
    price_max=100000,
    atr_ratio_min=0.003,
    volume_ma20_min=5000,
    amount_ma20_min=50_000_000,
    require_trend=True,
    max_picks=None,
)

BALANCED_RANKED_CONFIG = ScannerConfig(
    name="balanced_ranked",
    price_min=5000,
    price_max=200000,
    atr_ratio_min=0.002,
    volume_ma20_min=3000,
    amount_ma20_min=50_000_000,
    require_trend=False,
    max_picks=10,
)

AGGRESSIVE_RANKED_CONFIG = ScannerConfig(
    name="aggressive_ranked",
    price_min=5000,
    price_max=600000,
    atr_ratio_min=0.002,
    volume_ma20_min=3000,
    amount_ma20_min=50_000_000,
    require_trend=False,
    max_picks=15,
)

COMPARISON_CONFIGS = [
    CURRENT_STRICT_CONFIG,
    BALANCED_RANKED_CONFIG,
    AGGRESSIVE_RANKED_CONFIG,
]

DEFAULT_HISTORY_WINDOW = 20


def ensure_datetime_index(df):
    if isinstance(df.index, pd.DatetimeIndex):
        return df

    for col in ["date", "Date", "datetime", "Datetime", "time", "Time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df.set_index(col)
            return df

    return df


def load_data(code, data_dir, warn=True):
    file_path = data_dir / f"{code}.csv"

    if not file_path.exists():
        if warn:
            print(f"[WARN] {code} 파일 없음")
        return None

    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        df = ensure_datetime_index(df)
        return df
    except Exception as exc:
        if warn:
            print(f"[WARN] {code} 데이터 로드 실패: {exc}")
        return None


def filter_data_by_date(df, target_date):
    if target_date is None:
        return df

    if isinstance(df.index, pd.DatetimeIndex):
        # 데이터의 연도가 1900년인 경우(collector에서 날짜 결합 오류 등) 필터링을 건너뜁니다.
        if (df.index.year == 1900).any():
            return df
            
        return df[df.index.date == target_date.date()]

    return pd.DataFrame()


def calc_atr(df, length=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=length).mean()
    return atr


def build_indicators(df):
    if df is None or df.empty:
        return None

    df = df.copy()

    df["ATR"] = calc_atr(df, length=14)
    df["MA5"] = df["close"].rolling(5).mean()
    df["MA20"] = df["close"].rolling(20).mean()
    df["VOL_MA20"] = df["volume"].rolling(20).mean()
    df["AMOUNT"] = df["close"] * df["volume"]
    df["AMOUNT_MA20"] = df["AMOUNT"].rolling(20).mean()

    return df


def safe_float(value):
    if pd.isna(value):
        return None
    return float(value)


def classify_trend(ma5, ma20):
    if ma5 is None or ma20 is None:
        return "unknown"
    if ma5 > ma20:
        return "up"
    if ma5 < ma20:
        return "down"
    return "flat"


def calculate_candidate_score(candidate, config):
    atr_ratio = candidate["atr_ratio"]
    vol_ma20 = candidate["vol_ma20"]
    amount_ma20 = candidate["amount_ma20"]
    price = candidate["price"]
    ma5 = candidate["ma5"]
    ma20 = candidate["ma20"]

    if atr_ratio is None or vol_ma20 is None or amount_ma20 is None or price is None:
        return 0.0

    score = 0.0
    score += min(35.0, (atr_ratio / config.atr_ratio_min) * 17.5)
    score += min(25.0, (vol_ma20 / config.volume_ma20_min) * 8.0)
    score += min(25.0, (amount_ma20 / config.amount_ma20_min) * 8.0)

    if ma5 is not None and ma20 not in (None, 0):
        trend_ratio = (ma5 / ma20) - 1.0
        score += max(-15.0, min(15.0, trend_ratio * 1500.0))

    if price <= 100000:
        score += 5.0
    elif price <= config.price_max:
        score += 2.0

    return round(score, 2)


def evaluate_candidate(code, name, df, target_date=None, config=BALANCED_RANKED_CONFIG):
    candidate = {
        "code": code,
        "name": name,
        "eligible": False,
        "skip_reason": None,
        "fail_reasons": [],
        "soft_flags": [],
        "price": None,
        "atr": None,
        "atr_ratio": None,
        "ma5": None,
        "ma20": None,
        "ma_gap": None,
        "trend_state": "unknown",
        "vol_ma20": None,
        "amount_ma20": None,
        "score": 0.0,
    }

    prepared = build_indicators(df)
    if prepared is None:
        candidate["skip_reason"] = "no_data"
        return candidate

    filtered = filter_data_by_date(prepared, target_date)
    if filtered.empty:
        candidate["skip_reason"] = "date_missing"
        return candidate

    latest = filtered.iloc[-1]
    price = safe_float(latest["close"])
    atr = safe_float(latest["ATR"])
    ma5 = safe_float(latest["MA5"])
    ma20 = safe_float(latest["MA20"])
    vol_ma20 = safe_float(latest["VOL_MA20"])
    amount_ma20 = safe_float(latest["AMOUNT_MA20"])
    atr_ratio = None if price in (None, 0) or atr is None else atr / price
    ma_gap = None if ma5 is None or ma20 is None else ma5 - ma20

    candidate.update(
        {
            "price": price,
            "atr": atr,
            "atr_ratio": atr_ratio,
            "ma5": ma5,
            "ma20": ma20,
            "ma_gap": ma_gap,
            "trend_state": classify_trend(ma5, ma20),
            "vol_ma20": vol_ma20,
            "amount_ma20": amount_ma20,
        }
    )

    if atr is None or price is None:
        candidate["fail_reasons"].append("atr_or_price_nan")
        return candidate

    if price < config.price_min:
        candidate["fail_reasons"].append("price_floor")
    if price > config.price_max:
        candidate["fail_reasons"].append("price_ceiling")
    if atr_ratio is None or atr_ratio < config.atr_ratio_min:
        candidate["fail_reasons"].append("atr_ratio")
    if vol_ma20 is not None and vol_ma20 < config.volume_ma20_min:
        candidate["fail_reasons"].append("volume_ma20")
    if amount_ma20 is not None and amount_ma20 < config.amount_ma20_min:
        candidate["fail_reasons"].append("amount_ma20")

    if ma5 is not None and ma20 is not None and ma5 < ma20:
        if config.require_trend:
            candidate["fail_reasons"].append("ma_trend")
        else:
            candidate["soft_flags"].append("ma_trend")

    candidate["score"] = calculate_candidate_score(candidate, config)
    candidate["eligible"] = not candidate["fail_reasons"]

    return candidate


def format_metric(value, digits=0):
    if value is None:
        return "-"
    return f"{value:,.{digits}f}"


def summarize_candidates(candidates, selected_rows, skipped):
    fail_counts = {}
    soft_counts = {}
    for candidate in candidates:
        for reason in candidate["fail_reasons"]:
            fail_counts[reason] = fail_counts.get(reason, 0) + 1
        for reason in candidate["soft_flags"]:
            soft_counts[reason] = soft_counts.get(reason, 0) + 1

    return {
        "selected_count": len(selected_rows),
        "eligible_pool_count": sum(1 for row in candidates if row["eligible"]),
        "skipped": skipped,
        "fail_counts": dict(sorted(fail_counts.items(), key=lambda item: (-item[1], item[0]))),
        "soft_counts": dict(sorted(soft_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def render_ranked_csv(selected_rows):
    lines = ["rank,code,name,score,price,atr_ratio,vol_ma20,amount_ma20,trend_state,ma_gap"]
    for rank, row in enumerate(selected_rows, start=1):
        lines.append(
            ",".join(
                [
                    str(rank),
                    row["code"],
                    row["name"].replace(",", " "),
                    f"{row['score']:.2f}",
                    format_metric(row["price"], 0).replace(",", ""),
                    f"{row['atr_ratio']:.6f}" if row["atr_ratio"] is not None else "",
                    format_metric(row["vol_ma20"], 0).replace(",", ""),
                    format_metric(row["amount_ma20"], 0).replace(",", ""),
                    row["trend_state"],
                    f"{row['ma_gap']:.2f}" if row["ma_gap"] is not None else "",
                ]
            )
        )
    return "\n".join(lines)


def build_history_comparison(data_root, stock_list, history_window=DEFAULT_HISTORY_WINDOW):
    date_dirs = [d for d in data_root.iterdir() if d.is_dir() and d.name.isdigit()]
    date_dirs = sorted(date_dirs)[-history_window:]
    comparison_rows = []

    total_dates = len(date_dirs)
    print(f"\n📊 히스토리 비교 리포트 생성 중 ({total_dates}일치)...")

    for i, date_dir in enumerate(date_dirs, start=1):
        try:
            target_date = datetime.strptime(date_dir.name, "%Y%m%d")
        except ValueError:
            continue

        print(f"  [{i}/{total_dates}] {date_dir.name} 스캔 중...", end="\r", flush=True)

        strict = scan(
            date_dir,
            target_date=target_date,
            config=CURRENT_STRICT_CONFIG,
            return_details=True,
            verbose=False,
            stock_list=stock_list,
        )
        balanced = scan(
            date_dir,
            target_date=target_date,
            config=BALANCED_RANKED_CONFIG,
            return_details=True,
            verbose=False,
            stock_list=stock_list,
        )
        aggressive = scan(
            date_dir,
            target_date=target_date,
            config=AGGRESSIVE_RANKED_CONFIG,
            return_details=True,
            verbose=False,
            stock_list=stock_list,
        )

        comparison_rows.append(
            {
                "date": date_dir.name,
                "strict": strict["summary"]["selected_count"],
                "balanced_pool": strict["summary"]["selected_count"] if BALANCED_RANKED_CONFIG.max_picks is None else balanced["summary"]["eligible_pool_count"],
                "balanced_top": balanced["summary"]["selected_count"],
                "aggressive_pool": aggressive["summary"]["eligible_pool_count"],
                "aggressive_top": aggressive["summary"]["selected_count"],
            }
        )

    print(f"  히스토리 비교 완료 ({total_dates}일치)          ")
    return comparison_rows


def render_report(data_dir, target_date, config, scan_result, comparison_rows):
    target_label = target_date.strftime("%Y%m%d") if target_date else data_dir.name
    summary = scan_result["summary"]
    lines = [
        f"# Scanner Report - {target_label}",
        "",
        "## Active Scanner Config",
        "",
        f"- config: {config.name}",
        f"- price range: {config.price_min:,} ~ {config.price_max:,}",
        f"- ATR/price min: {config.atr_ratio_min:.4f}",
        f"- volume MA20 min: {config.volume_ma20_min:,}",
        f"- amount MA20 min: {config.amount_ma20_min:,}",
        f"- require trend: {config.require_trend}",
        f"- max picks: {config.max_picks}",
        "",
        "## Summary",
        "",
        f"- eligible pool: {summary['eligible_pool_count']}",
        f"- selected picks: {summary['selected_count']}",
        f"- skipped: {summary['skipped']}",
        "",
        "## Ranked Picks",
        "",
        "| rank | code | name | score | price | ATR/price | vol MA20 | amount MA20 | trend | ma gap |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]

    for rank, row in enumerate(scan_result["selected_rows"], start=1):
        lines.append(
            "| {rank} | {code} | {name} | {score:.2f} | {price} | {atr_ratio:.4%} | {vol_ma20} | {amount_ma20} | {trend} | {ma_gap} |".format(
                rank=rank,
                code=row["code"],
                name=row["name"],
                score=row["score"],
                price=format_metric(row["price"], 0),
                atr_ratio=row["atr_ratio"] or 0.0,
                vol_ma20=format_metric(row["vol_ma20"], 0),
                amount_ma20=format_metric(row["amount_ma20"], 0),
                trend=row["trend_state"],
                ma_gap=format_metric(row["ma_gap"], 2),
            )
        )

    lines.extend([
        "",
        "## Filter Counts",
        "",
        "| reason | count |",
        "| --- | ---: |",
    ])
    for reason, count in summary["fail_counts"].items():
        lines.append(f"| {reason} | {count} |")

    if summary["soft_counts"]:
        lines.extend([
            "",
            "## Soft Flags",
            "",
            "| flag | count |",
            "| --- | ---: |",
        ])
        for reason, count in summary["soft_counts"].items():
            lines.append(f"| {reason} | {count} |")

    if comparison_rows:
        lines.extend([
            "",
            f"## Recent {len(comparison_rows)}-Day Comparison",
            "",
            "| date | strict | balanced pool | balanced top | aggressive pool | aggressive top |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in comparison_rows:
            lines.append(
                "| {date} | {strict} | {balanced_pool} | {balanced_top} | {aggressive_pool} | {aggressive_top} |".format(**row)
            )

    return "\n".join(lines) + "\n"


def get_latest_data_dir(base_path="data"):
    base = Path(base_path)

    if not base.exists():
        raise SystemExit("❌ data 폴더가 없습니다. collector 먼저 실행하세요.")

    dirs = [d for d in base.iterdir() if d.is_dir()]
    if not dirs:
        raise SystemExit("❌ 데이터 폴더가 비어 있습니다.")

    latest_dir = sorted(dirs)[-1]
    return latest_dir


def load_symbols():
    symbols_path = Path("symbols.csv")

    if not symbols_path.exists():
        raise SystemExit("❌ symbols.csv 파일이 없습니다.")

    df = pd.read_csv(symbols_path)

    if "code" not in df.columns:
        raise SystemExit("❌ symbols.csv에 'code' 컬럼 필요")

    # code와 name 컬럼을 함께 반환 (name이 없으면 빈 문자열)
    name_col = "name" if "name" in df.columns else df.columns[1] if len(df.columns) > 1 else None
    
    if name_col:
        return list(zip(df["code"].astype(str).str.zfill(6), df[name_col].astype(str)))
    return [(c, "") for c in df["code"].astype(str).str.zfill(6)]


def scan(
    data_dir,
    target_date=None,
    config=BALANCED_RANKED_CONFIG,
    return_details=False,
    verbose=True,
    stock_list=None,
):
    stock_list = stock_list or load_symbols()

    total = len(stock_list)
    if verbose:
        print(f"🔍 총 {total} 종목 스캔 시작 (폴더: {data_dir}, config: {config.name})\n")

    candidates = []
    skipped = 0

    for idx, (code, name) in enumerate(stock_list, start=1):
        if verbose:
            print(f"[{idx}/{total}] {code} 검사중...")

        df = load_data(code, data_dir, warn=verbose)

        if df is None:
            skipped += 1
            continue
        if len(df) < 50:
            skipped += 1
            if verbose:
                print(f"[{idx}/{total}] {code} ⏭ 스킵 (데이터 부족)")
            continue

        candidate = evaluate_candidate(code, name, df, target_date=target_date, config=config)
        candidates.append(candidate)

        if verbose:
            if candidate["eligible"]:
                print(f"[{idx}/{total}] {code} ✅ 후보(score={candidate['score']:.2f})")
            else:
                reasons = candidate["fail_reasons"] or [candidate["skip_reason"] or "unknown"]
                print(f"[{idx}/{total}] {code} ❌ 탈락 ({', '.join(reasons)})")

    eligible_rows = [row for row in candidates if row["eligible"]]
    eligible_rows.sort(
        key=lambda row: (row["score"], row["amount_ma20"] or 0.0, row["atr_ratio"] or 0.0),
        reverse=True,
    )

    selected_rows = eligible_rows
    if config.max_picks is not None:
        selected_rows = eligible_rows[:config.max_picks]

    selected = [f"{row['code']},{row['name']}" for row in selected_rows]
    summary = summarize_candidates(candidates, selected_rows, skipped)

    if verbose:
        print("\n========== 스캔 완료 ==========")
        print(f"총 종목: {total}")
        print(f"적격 풀: {summary['eligible_pool_count']}")
        print(f"최종 선정: {summary['selected_count']}")
        print(f"스킵: {skipped}")

    if return_details:
        return {
            "selected": selected,
            "selected_rows": selected_rows,
            "eligible_rows": eligible_rows,
            "summary": summary,
            "config": config,
        }

    return selected


def parse_args():
    parser = argparse.ArgumentParser(description="Scanner (collector 연동 버전)")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="기준 날짜 (YYYYMMDD)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="데이터 폴더 (예: data/20260328)"
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=DEFAULT_HISTORY_WINDOW,
        help="비교 리포트에 포함할 최근 거래일 수"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 📅 날짜 파싱
    target_date = None
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y%m%d")
        except ValueError:
            raise SystemExit("날짜 형식은 YYYYMMDD")

    # 📂 데이터 폴더 선택 로직 개선
    data_dir = None
    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif target_date:
        # 입력된 날짜(YYYY-MM-DD)를 YYYYMMDD 폴더 형식으로 변환하여 확인
        folder_name = target_date.strftime("%Y%m%d")
        data_dir = Path("data") / folder_name

    if data_dir is None or not data_dir.exists():
        print(f"[INFO] 지정된 날짜 폴더가 없어 최신 폴더를 검색합니다.")
        data_dir = get_latest_data_dir()

    stock_list = load_symbols()
    scan_result = scan(
        data_dir,
        target_date,
        config=BALANCED_RANKED_CONFIG,
        return_details=True,
        stock_list=stock_list,
    )
    picks = scan_result["selected"]

    comparison_rows = []
    if args.history_window > 0:
        comparison_rows = build_history_comparison(Path("data"), stock_list, history_window=args.history_window)

    # 💾 결과 저장 (trader.py 연동용)
    if picks:
        picks_file = data_dir / "picks.txt"
        with open(picks_file, "w") as f:
            f.write("\n".join(picks))
        print(f"✅ 추천 종목 리스트가 저장되었습니다: {picks_file}")

    ranked_file = data_dir / "picks_ranked.csv"
    ranked_file.write_text(render_ranked_csv(scan_result["selected_rows"]), encoding="utf-8")
    print(f"✅ 점수 기반 랭킹 파일이 저장되었습니다: {ranked_file}")

    report_file = data_dir / "scanner_report.md"
    report_file.write_text(
        render_report(data_dir, target_date, BALANCED_RANKED_CONFIG, scan_result, comparison_rows),
        encoding="utf-8",
    )
    print(f"✅ 스캐너 리포트가 저장되었습니다: {report_file}")

    if target_date:
        print(f"\n🔥 {target_date.strftime('%Y%m%d')} 기준 추천 종목:", picks)
    else:
        print("\n🔥 최신 데이터 기준 추천 종목:", picks)

