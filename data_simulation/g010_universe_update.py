"""
g010_universe_update.py

g004_universe_symbols_master.txt(수집/스캔 대상 종목 목록)를 한국투자 종목마스터(kospi_code.mst /
kosdaq_code.mst, API 키 불필요, 매일 갱신)와 맞춘다.

- 신규상장: 목록에 없고, 목록 안 종목 중 가장 최근 상장일 이후 상장된 종목을 추가
  (주권/외국주권/DR/리츠만, 스팩/ETF/ETN 제외). 삼성전자처럼 오래전 상장인데 목록에 없는
  종목은 의도적 제외로 보고 추가하지 않음. 첫 실행 등 기준을 바꾸려면 --new-since.
- 상장폐지: 두 마스터 어디에도 없는 종목 줄을 삭제.
- 종목명/시장(코스닥->코스피 이전상장 등) 변경 반영.
- 위험종목은 "# " 주석 처리하고 사유를 붙임:
      # 002360,SH에너지화학,KOSPI  # auto: 투자경고, 단기과열지정예고
  사유: 거래정지, 정리매매, 관리종목, 투자주의/경고/위험, 투자경고·위험 지정예고,
  단기과열(지정예고/지정/연장), 이상급등, 공매도과열, 불성실공시, 투자주의환기(코스닥),
  신규상장 후 NEW_LISTING_COOLDOWN_DAYS일 이내.
  위험이 해제되면 "# auto:" 줄은 자동으로 주석이 풀린다.
- 사람이 직접 "#"로 막은 줄("# auto:" 표시가 없는 주석)은 절대 건드리지 않는다
  (해제/삭제/이름변경 없음). 다시 쓰려면 직접 "#"를 지우면 된다.

Run examples:
- python g010_universe_update.py --dry-run          (변경 내역만 출력)
- python g010_universe_update.py                    (g004 파일 갱신)
- python g010_universe_update.py --new-since 20260601

권장 실행 시점: g001 수집 전(마스터 파일은 매일 새벽 갱신됨).

Update log format (append only):
- [YYYY-MM-DD] type=<feat|fix|refactor|perf|docs|chore|test> owner=<name>
    summary: <what changed and why>
    impact: <live|sim|scanner|collector|common|docs>
    compatibility: <backward-compatible|breaking>

Update log:
- [2026-10-09] type=feat owner=claude
    summary: 신규 - 종목마스터 기반 유니버스 갱신(신규상장 추가/상장폐지 삭제/종목명·시장 변경/
      위험종목 자동 주석). 필드 배치는 open-trading-api/stocks_info/kis_kospi_code_mst.py,
      kis_kosdaq_code_mst.py 및 종목마스터정보(코스피/코스닥).h 기준(라이브러리는 import 시
      다운로드를 실행해서 import하지 않고 배치만 옮김).
    impact: collector, scanner
    compatibility: backward-compatible (파일 형식 code,name,market 유지; 주석 줄은 g001/g002가
      comment="#"로 건너뜀)
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import accumulate
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SYMBOLS_FILE = SCRIPT_DIR / "g004_universe_symbols_master.txt"

MASTER_URL = "https://new.real.download.dws.co.kr/common/master/{market}_code.mst.zip"
DOWNLOAD_TIMEOUT = (10, 60)
DOWNLOAD_RETRIES = 3
# Abort without writing if a master looks truncated (normal: KOSPI ~900 / KOSDAQ ~1800 주권).
MIN_STOCK_ROWS = {"KOSPI": 500, "KOSDAQ": 1000}

NEW_LISTING_GROUPS = {"ST", "FS", "DR", "RT"}  # 주권, 외국주권, 주식예탁증서, 리츠
NEW_LISTING_COOLDOWN_DAYS = 7                  # 상장 직후 급등락 구간(달력일)
AUTO_TAG = "# auto:"

# Fixed-width tail layout after 단축코드(9)+표준코드(12)+한글명. Widths copied from
# open-trading-api/stocks_info/kis_{kospi,kosdaq}_code_mst.py; only the fields used here are named.
_KOSPI_WIDTHS = [2, 1, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 9, 5, 5, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 3, 1, 3, 12, 12, 8,
                 15, 21, 2, 7, 1, 1, 1, 1, 1, 9, 9, 9, 5, 9, 8, 9, 3, 1, 1, 1]
_KOSPI_FIELDS = {0: "grp", 19: "spac", 22: "short_over", 34: "halt", 35: "liquidation", 36: "admin",
                 37: "warn", 38: "warn_pre", 39: "unfaithful", 49: "listed", 55: "short_hot", 56: "runup"}
_KOSDAQ_WIDTHS = [2, 1, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 9, 5, 5, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 3, 1, 3, 12, 12, 8, 15, 21, 2, 7, 1,
                  1, 1, 1, 9, 9, 9, 5, 9, 8, 9, 3, 1, 1, 1]
_KOSDAQ_FIELDS = {0: "grp", 14: "spac", 17: "short_over", 20: "caution_alert", 29: "halt",
                  30: "liquidation", 31: "admin", 32: "warn", 33: "warn_pre", 34: "unfaithful",
                  44: "listed", 50: "short_hot", 51: "runup"}
MASTER_LAYOUTS = {"KOSPI": (_KOSPI_WIDTHS, _KOSPI_FIELDS), "KOSDAQ": (_KOSDAQ_WIDTHS, _KOSDAQ_FIELDS)}

WARN_LABELS = {"01": "투자주의", "02": "투자경고", "03": "투자위험"}
SHORT_OVER_LABELS = {"1": "단기과열지정예고", "2": "단기과열지정", "3": "단기과열지정연장"}
FLAG_LABELS = [  # (field, label) - "Y" means risky
    ("halt", "거래정지"), ("liquidation", "정리매매"), ("admin", "관리종목"),
    ("warn_pre", "투자경고·위험 지정예고"), ("runup", "이상급등"), ("short_hot", "공매도과열"),
    ("unfaithful", "불성실공시"), ("caution_alert", "투자주의환기"),
]

CODE_RE = re.compile(r"([0-9A-Z]{6})$")


# ---------------------------------------------------------------------------
# KIS master
# ---------------------------------------------------------------------------

def download_master(market: str) -> bytes:
    url = MASTER_URL.format(market=market.lower())
    last_exc: Exception | None = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            res = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
            res.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
                return zf.read(f"{market.lower()}_code.mst")
        except (requests.RequestException, zipfile.BadZipFile, KeyError) as exc:
            last_exc = exc
            print(f"[WARN] {market} master download failed ({attempt}/{DOWNLOAD_RETRIES}): {exc}")
            time.sleep(2 * attempt)
    raise SystemExit(f"[ERROR] {market} master download failed: {last_exc}")


def parse_master(market: str, raw: bytes) -> dict[str, dict]:
    widths, fields = MASTER_LAYOUTS[market]
    tail = sum(widths)
    bounds = [0, *accumulate(widths)]
    rows: dict[str, dict] = {}
    for line in raw.decode("cp949", errors="replace").splitlines():
        if len(line) <= tail + 21:
            continue
        head, body = line[:-tail], line[-tail:]
        rec = {name: body[bounds[i]:bounds[i + 1]].strip() for i, name in fields.items()}
        rec["code"] = head[0:9].strip()
        rec["name"] = head[21:].strip()
        rec["market"] = market
        if rec["code"]:
            rows[rec["code"]] = rec
    n_stock = sum(1 for r in rows.values() if r["grp"] == "ST")
    if n_stock < MIN_STOCK_ROWS[market]:
        raise SystemExit(f"[ERROR] {market} master looks truncated ({n_stock} 주권 rows) - nothing written")
    return rows


def risk_reasons(rec: dict, today: datetime) -> list[str]:
    reasons: list[str] = []
    for fld, label in FLAG_LABELS[:3]:
        if rec.get(fld) == "Y":
            reasons.append(label)
    if rec.get("warn") in WARN_LABELS:
        reasons.append(WARN_LABELS[rec["warn"]])
    if rec.get("short_over") in SHORT_OVER_LABELS:
        reasons.append(SHORT_OVER_LABELS[rec["short_over"]])
    for fld, label in FLAG_LABELS[3:]:
        if rec.get(fld) == "Y":
            reasons.append(label)
    try:
        listed = datetime.strptime(rec.get("listed", ""), "%Y%m%d")
        if today - listed < timedelta(days=NEW_LISTING_COOLDOWN_DAYS):
            reasons.append(f"신규상장({listed:%m/%d})")
    except ValueError:
        pass
    return reasons


# ---------------------------------------------------------------------------
# Universe file
# ---------------------------------------------------------------------------

@dataclass
class Line:
    kind: str                 # header | blank | active | auto | manual | other
    raw: str
    code: str = ""
    name: str = ""
    market: str = ""
    reasons: list[str] = field(default_factory=list)

    def render(self) -> str:
        if self.kind in ("active", "auto"):
            body = f"{self.code},{self.name},{self.market}"
            return body if self.kind == "active" else f"# {body}  {AUTO_TAG} {', '.join(self.reasons)}"
        return self.raw


def _split_entry(text: str) -> tuple[str, str, str] | None:
    parts = [p.strip() for p in text.split(",")]
    m = CODE_RE.search(parts[0]) if parts else None
    if not m:
        return None
    return m.group(1), (parts[1] if len(parts) > 1 else ""), (parts[2] if len(parts) > 2 else "")


def parse_universe(path: Path) -> tuple[list[Line], list[str]]:
    lines: list[Line] = []
    fixes: list[str] = []
    for i, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines()):
        stripped = raw.strip()
        if i == 0 and stripped.startswith("code"):
            lines.append(Line("header", raw))
        elif not stripped:
            lines.append(Line("blank", raw))
        elif stripped.startswith("#"):
            body = stripped.lstrip("#").strip()
            if AUTO_TAG in stripped:
                entry_text, reason_text = body.split(AUTO_TAG, 1)
                entry = _split_entry(entry_text.strip())
                if entry:
                    reasons = [r.strip() for r in reason_text.split(",") if r.strip()]
                    lines.append(Line("auto", raw, *entry, reasons=reasons))
                    continue
            entry = _split_entry(body)
            lines.append(Line("manual", raw, *(entry or ("", "", ""))))
        else:
            entry = _split_entry(stripped)
            if entry is None:
                lines.append(Line("other", raw))
                fixes.append(f"알 수 없는 줄 유지: {raw}")
                continue
            if not stripped.startswith(entry[0]):
                fixes.append(f"깨진 코드 정리: {raw} -> {entry[0]}")
            lines.append(Line("active", raw, *entry))
    if not lines or lines[0].kind != "header":
        lines.insert(0, Line("header", "code,name,market"))
    return lines, fixes


def update_universe(lines: list[Line], master: dict[str, dict], today: datetime, new_since: str | None) -> dict:
    report: dict[str, list] = {k: [] for k in
                               ("added", "removed", "renamed", "market", "commented", "uncommented",
                                "reasons_changed", "duplicate", "fixes")}
    kept: list[Line] = []
    seen: set[str] = set()
    for ln in lines:
        if ln.kind not in ("active", "auto"):
            kept.append(ln)
            continue
        if ln.code in seen:
            report["duplicate"].append(f"{ln.code},{ln.name}")
            continue
        seen.add(ln.code)
        rec = master.get(ln.code)
        if rec is None:
            report["removed"].append(f"{ln.code},{ln.name},{ln.market}")
            continue
        if rec["name"] and rec["name"] != ln.name:
            report["renamed"].append(f"{ln.code} {ln.name} -> {rec['name']}")
            ln.name = rec["name"]
        if rec["market"] != ln.market:
            report["market"].append(f"{ln.code},{ln.name} {ln.market or '-'} -> {rec['market']}")
            ln.market = rec["market"]
        reasons = risk_reasons(rec, today)
        if reasons and ln.kind == "active":
            report["commented"].append(f"{ln.code},{ln.name}: {', '.join(reasons)}")
        elif not reasons and ln.kind == "auto":
            report["uncommented"].append(f"{ln.code},{ln.name} (해제: {', '.join(ln.reasons)})")
        elif reasons and reasons != ln.reasons:
            report["reasons_changed"].append(f"{ln.code},{ln.name}: {', '.join(ln.reasons)} -> {', '.join(reasons)}")
        ln.kind = "auto" if reasons else "active"
        ln.reasons = reasons
        kept.append(ln)

    # New listings: newer than the most recently listed stock already in the file. Codes that
    # appear anywhere in the file (incl. manual "#" lines) are never re-added.
    present = seen | {ln.code for ln in kept if ln.kind == "manual" and ln.code}
    baseline = new_since or max(
        (master[c]["listed"] for c in seen if c in master and master[c]["grp"] in NEW_LISTING_GROUPS),
        default="99999999",
    )
    new_lines: list[Line] = []
    for rec in sorted(master.values(), key=lambda r: (r["listed"], r["code"])):
        if (rec["code"] in present or rec["grp"] not in NEW_LISTING_GROUPS or rec.get("spac") == "Y"
                or rec["listed"] <= baseline or len(rec["code"]) != 6):
            continue
        reasons = risk_reasons(rec, today)
        new_lines.append(Line("auto" if reasons else "active", "", rec["code"], rec["name"], rec["market"], reasons))
        report["added"].append(f"{rec['code']},{rec['name']},{rec['market']} (상장 {rec['listed']})"
                               + (f" -> 주석: {', '.join(reasons)}" if reasons else ""))
    last_data = max((i for i, ln in enumerate(kept) if ln.kind in ("active", "auto", "header")), default=0)
    kept[last_data + 1:last_data + 1] = new_lines

    lines[:] = kept
    report["baseline"] = baseline
    return report


def write_atomic(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync g004 universe with KIS stock master (listing/delisting/risk flags)")
    parser.add_argument("--symbols-file", default=str(DEFAULT_SYMBOLS_FILE))
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    parser.add_argument("--new-since", default=None,
                        help="YYYYMMDD: add stocks listed after this date (default: newest listing already in file)")
    args = parser.parse_args()

    if args.new_since:
        datetime.strptime(args.new_since, "%Y%m%d")
    path = Path(args.symbols_file)
    if not path.is_file():
        raise SystemExit(f"symbols file not found: {path}")

    master: dict[str, dict] = {}
    for market in ("KOSPI", "KOSDAQ"):
        rows = parse_master(market, download_master(market))
        print(f"[INFO] {market} master: {len(rows)} rows ({sum(1 for r in rows.values() if r['grp'] == 'ST')} 주권)")
        master.update(rows)

    lines, fixes = parse_universe(path)
    today = datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)
    report = update_universe(lines, master, today, args.new_since)
    report["fixes"] = fixes

    titles = [("added", "신규상장 추가"), ("removed", "상장폐지(마스터 없음) 삭제"), ("renamed", "종목명 변경"),
              ("market", "시장 변경"), ("commented", "위험종목 주석 처리"), ("uncommented", "위험 해제 - 주석 해제"),
              ("reasons_changed", "주석 사유 변경"), ("duplicate", "중복 줄 삭제"), ("fixes", "기타 정리")]
    print(f"[INFO] 신규상장 기준일: {report['baseline']} 이후 상장")
    for key, title in titles:
        items = report[key]
        print(f"\n[{title}] {len(items)}")
        for item in items:
            print(f"  {item}")

    n_active = sum(1 for ln in lines if ln.kind == "active")
    n_auto = sum(1 for ln in lines if ln.kind == "auto")
    n_manual = sum(1 for ln in lines if ln.kind == "manual")
    print(f"\n[INFO] 결과: 수집 대상 {n_active} / 자동 주석 {n_auto} / 수동 주석 {n_manual}")

    text = "\n".join(ln.render() for ln in lines) + "\n"
    if args.dry_run:
        print("[INFO] --dry-run: 파일을 쓰지 않음")
    elif text == path.read_text(encoding="utf-8-sig"):
        print("[INFO] 변경 없음")
    else:
        write_atomic(path, text)
        print(f"[INFO] saved: {path}")


if __name__ == "__main__":
    main()
