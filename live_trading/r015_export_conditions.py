# -*- coding: utf-8 -*-
"""R015 - 매수/매도 조건 목록을 JSON으로 내보낸다 (조건 다이어그램 뷰어용, 읽기 전용).

실행: ~/venv/bin/python live_trading/r015_export_conditions.py [--out path.json]
출력: {"generated_at", "entry_trigger_mode", "trading_strategy_mode", "buy": [...], "sell": [...]}
  - buy/sell 항목: {"order": 평가 순서(0부터), "no": 번호, "func": 함수명, "stage": STATE|TRIGGER|CONTEXT|PRE_ORDER / GUARD|UPDATE|EXIT|STATE,
                   "title", "description", "defines": [r001 상수명...], "values": {상수명: 값}}
  - 브로커 import 없음. r005/r006의 describe_* 와 active_* 를 그대로 쓰므로 코드가 바뀌면 출력도 바뀐다.

Update log:
- [2026-11-04] type=feat owner=claude
    summary: 윈도우 조건 뷰어(ssh로 로그 수집) 구현 가이드용 - 조건 순서/설명/관련 상수 값을 JSON으로.
    impact: tooling
    compatibility: backward-compatible
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import contextlib  # noqa: E402
import io  # noqa: E402

# r001 import가 stdout에 "[INFO] Applied risk profile ..."를 찍으므로 JSON만 나가도록 import 동안 stdout을 막는다.
with contextlib.redirect_stdout(io.StringIO()):
    import r001_define_config as r001  # noqa: E402
    import r005_buy_conditions as r005  # noqa: E402
    import r006_sell_conditions as r006  # noqa: E402


def _value(name: str):
    v = getattr(r001, name, None)
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, (tuple, list, set, frozenset)):
        return [x if isinstance(x, (int, float, str, bool)) else str(x) for x in v]
    if isinstance(v, dict):
        return {str(k): (x if isinstance(x, (int, float, str, bool)) else str(x)) for k, x in v.items()}
    return str(v)


def _rows(conditions, describe_rows):
    desc = {d["func"]: d for d in describe_rows}
    out = []
    for order, c in enumerate(conditions):
        d = desc.get(c.func_name, {})
        defines = list(getattr(c, "related_defines", ()) or ())
        out.append({
            "order": order, "no": c.no, "func": c.func_name, "stage": getattr(c, "stage", getattr(c, "kind", "")), "title": c.title,
            "description": d.get("description", getattr(c, "description", "")),
            "defines": defines, "values": {n: _value(n) for n in defines},
        })
    return out


def build() -> dict:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "entry_trigger_mode": str(getattr(r001, "ENTRY_TRIGGER_MODE", "")),
        "trading_strategy_mode": str(getattr(r001, "TRADING_STRATEGY_MODE", "")),
        "buy": _rows(r005.active_buy_conditions(), r005.describe_buy_conditions()),
        "sell": _rows(r006.active_sell_conditions(), r006.describe_sell_conditions()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="export r005/r006 condition lists as JSON")
    ap.add_argument("--out", default="", help="write to this file instead of stdout")
    args = ap.parse_args()
    text = json.dumps(build(), ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
