"""
현재가 조회 스크립트.

사용 예시:
  python r011_current_value.py --code 005930
"""

import argparse
import sys
from pathlib import Path


def _extract_current_price(row: dict) -> str:
    # API 스펙/버전별 키 차이를 흡수하기 위한 후보 키 목록
    for key in ("stck_prpr", "stck_clpr", "cur_prc", "prpr"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return "N/A"


def main() -> None:
    parser = argparse.ArgumentParser(description="종목코드로 현재가를 조회합니다")
    parser.add_argument("--code", type=str, required=True, help="종목코드 (예: 005930)")
    args = parser.parse_args()

    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parents[1]

    sys.path.insert(0, str(project_root / "examples_llm"))
    sys.path.insert(0, str(project_root / "examples_user" / "domestic_stock"))

    import kis_auth as ka
    from domestic_stock_functions import inquire_price

    code = args.code.strip().zfill(6)

    ka.auth()
    df = inquire_price(
        env_dv="real",
        fid_cond_mrkt_div_code="J",
        fid_input_iscd=code,
    )

    if df is None or df.empty:
        print(f"[조회 실패] {code} 현재가 데이터를 가져오지 못했습니다.")
        return

    row = df.iloc[0].to_dict()
    current_price = _extract_current_price(row)
    name = row.get("hts_kor_isnm") or row.get("prdt_name") or ""
    change = row.get("prdy_vrss") or ""
    change_rate = row.get("prdy_ctrt") or ""

    print(f"종목코드: {code}")
    if name:
        print(f"종목명: {name}")
    print(f"현재가: {current_price}")
    if change != "":
        print(f"전일대비: {change}")
    if change_rate != "":
        print(f"등락률: {change_rate}%")


if __name__ == "__main__":
    main()
