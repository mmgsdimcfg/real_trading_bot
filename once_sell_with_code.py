"""
일회용 매도 스크립트 (One-time Sell Script)

한국투자 API를 사용하여 특정 종목코드로 시장가 매도를 실행하는 스크립트입니다.
모의투자와 실전투자를 지원합니다.

[사용 예제]

1. 기본 매도 (시장가로 삼성전자 100주 매도)
   python one_sell_with_code.py --code 005930 --count 100

2. 다른 종목 매도 (LG전자 50주 매도)
   python one_sell_with_code.py --code 066570 --count 50

3. 코스닥 종목 매도 (셀트리온헬스케어 200주 매도)
   python one_sell_with_code.py --code 091990 --count 200

[주의사항]
- 코드 앞에 0이 필요한 경우 자동 처리됩니다 (예: 5930 → 005930)
- 기본값은 실전투자("real")입니다. 모의투자는 main 함수의 env_dv="demo"으로 변경하세요.
- 실전투자 시 실제 매매가 발생하니 주의하세요.
- 보유하고 있는 주식만 매도할 수 있습니다.
"""

import sys
import argparse
from pathlib import Path

# 프로젝트 루트 및 모듈 경로 설정
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[1]

sys.path.insert(0, str(project_root / "examples_llm"))
sys.path.insert(0, str(project_root / "examples_user" / "domestic_stock"))

import kis_auth as ka
from domestic_stock_functions import order_cash

def main():
    parser = argparse.ArgumentParser(description="Koreainvest API 즉시 매도 스크립트")
    parser.add_argument("--code", type=str, required=True, help="종목코드 (예: 005930)")
    parser.add_argument("--count", type=int, required=True, help="매도 수량")
    args = parser.parse_args()

    # 1. API 인증 및 계좌 정보 로드
    ka.auth()
    trenv = ka.getTREnv()
    cano = trenv.my_acct
    acnt_prdt_cd = trenv.my_prod
    
    code = args.code.zfill(6)
    qty = str(args.count)

    print(f"💰 [매도 시도] {code} | 수량: {qty}주 | 시장가(01)")

    # 2. 주문 실행 (시장가 매도)
    result = order_cash(
        env_dv="real",  # 실전투자: "real", 모의투자: "demo"
        ord_dv="sell",
        cano=cano,
        acnt_prdt_cd=acnt_prdt_cd,
        pdno=code,
        ord_dvsn="01",  # 01: 시장가
        ord_qty=qty,
        ord_unpr="0",   # 시장가는 가격 0
        excg_id_dvsn_cd="KRX"
    )

    if result is not None and not result.empty:
        print(f"✅ [매도 완료] 주문 성공")
    else:
        print(f"❌ [매도 실패] 주문 처리 중 오류가 발생했거나 결과가 없습니다.")

if __name__ == "__main__":
    main()