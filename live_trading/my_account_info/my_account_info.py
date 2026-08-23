# -*- coding: utf-8 -*-
"""내 계좌 정보 조회 스크립트 - 잔액, 보유종목, 수익률"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd

# 절대경로 설정
project_root = Path(os.environ.get("OPEN_TRADING_API_ROOT", str(Path.home() / "git" / "open-trading-api")))
sys.path.insert(0, str(project_root / "examples_llm"))

import kis_auth as ka
# 국내주식 함수들 import - 올바른 경로로 수정
sys.path.insert(0, str(project_root / "examples_user" / "domestic_stock"))
from domestic_stock_functions import inquire_account_balance, inquire_balance_rlz_pl

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """
    계좌 정보 조회:
    1. 계좌 자산 현황 (총 자산, 예수금 등)
    2. 보유 종목 정보 (종목명, 보유수량, 매수가, 현재가, 평가손익, 수익률)
    """
    
    # 설정
    env_dv = "demo"  # 실전: "real", 모의: "demo"
    
    # 토큰 발급
    logger.info("토큰 발급 중...")
    ka.auth()
    logger.info("토큰 발급 완료")
    
    # kis_auth의 설정에서 계좌정보 읽기
    # auth() 호출 후 getTREnv()로 계좌 정보 가져오기
    trenv = ka.getTREnv()
    cano = trenv.my_acct  # 계좌번호 (앞 8자리)
    acnt_prdt_cd = trenv.my_prod  # 계좌상품코드 (뒤 2자리)
    
    if not cano or not acnt_prdt_cd:
        logger.error("계좌 정보가 설정되지 않았습니다. kis_devlp.yaml을 확인하세요.")
        return
    
    logger.info(f"조회 계좌: {cano}-{acnt_prdt_cd}")
    
    # =====================================================================
    # 1. 계좌 자산 현황 조회 (inquire_account_balance)
    # =====================================================================
    logger.info("\n[1] 계좌 자산 현황 조회 중...")
    try:
        df_asset_overview, df_asset_detail = inquire_account_balance(
            cano=cano,
            acnt_prdt_cd=acnt_prdt_cd,
            inqr_dvsn_1="",
            bspr_bf_dt_aply_yn=""
        )
        
        if not df_asset_overview.empty:
            print("\n" + "="*80)
            print("【 계좌 자산 현황 】")
            print("="*80)
            # 주요 컬럼만 출력
            asset_cols = [col for col in df_asset_overview.columns 
                         if any(x in col.lower() for x in ['asst', 'deposit', 'evlu', 'marg', 'prcs'])]
            if asset_cols:
                print(df_asset_overview[asset_cols])
            else:
                print(df_asset_overview)
                
        if not df_asset_detail.empty:
            print("\n" + "="*80)
            print("【 계좌 상세 정보 】")
            print("="*80)
            print(df_asset_detail)
            
    except Exception as e:
        logger.warning(f"계좌 자산 조회 실패: {str(e)}")
    
    # =====================================================================
    # 2. 보유 종목 및 손익 정보 조회 (inquire_balance_rlz_pl)
    # =====================================================================
    logger.info("\n[2] 보유 종목 및 손익 정보 조회 중...")
    try:
        df_holdings, df_summary = inquire_balance_rlz_pl(
            cano=cano,
            acnt_prdt_cd=acnt_prdt_cd,
            afhr_flpr_yn="N",  # 시간외단일가여부
            inqr_dvsn="02",  # 조회구분 (02: 종목별)
            unpr_dvsn="01",  # 단가구분
            fund_sttl_icld_yn="N",  # 펀드결제포함여부
            fncg_amt_auto_rdpt_yn="N",  # 융자금액자동상환여부
            prcs_dvsn="00"  # 처리구분 (00: 전일매매포함)
        )
        
        if not df_holdings.empty:
            # 보유수량이 0인 종목 필터링 (매도 완료 종목 제외)
            df_holdings = df_holdings[pd.to_numeric(df_holdings['hldg_qty'], errors='coerce') > 0].copy()
            
        if not df_holdings.empty:
            print("\n" + "="*80)
            print("【 보유 종목 및 손익 정보 】")
            print("="*80)
            
            # 선택할 컬럼들
            display_cols = []
            col_mapping = {
                'hldg_qty': '보유수량',
                'pchs_avg_pric': '매수평균가',
                'prpr': '현재가',
                'evlu_pfls_amt': '평가손익',
                'evlu_pfls_rt': '수익률(%)',
                'prdt_name': '종목명',
                'pdno': '종목코드'
            }
            
            # 사용 가능한 컬럼만 수집
            for col, label in col_mapping.items():
                if col in df_holdings.columns:
                    display_cols.append(col)
            
            if display_cols:
                df_display = df_holdings[display_cols].copy()
                # 컬럼명 변경
                rename_dict = {col: col_mapping.get(col, col) for col in display_cols}
                df_display = df_display.rename(columns=rename_dict)
                print(df_display.to_string(index=False))
            else:
                print("표시할 컬럼이 없습니다. 전체 컬럼:")
                print(df_holdings.columns.tolist())
                print(df_holdings)
            
            # 통계 정보
            print(f"\n【 보유 종목 통계 】")
            print(f"보유 종목 수: {len(df_holdings)}")
            
            # numeric columns만 처리
            numeric_cols = df_holdings.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if 'qty' in col.lower():
                    print(f"총 보유수량: {df_holdings[col].sum():,.0f}")
                elif 'amt' in col.lower() and 'pfls' in col.lower():
                    total = df_holdings[col].sum()
                    print(f"총 {col}: {total:,.0f}" if pd.notna(total) else f"{col}: N/A")
        
        if not df_summary.empty:
            # 계좌 총괄 요약 계산 및 출력
            summary = df_summary.iloc[0]
            
            # 데이터 추출 (안전을 위해 numeric 변환)
            tot_evlu_amt = pd.to_numeric(summary.get('tot_evlu_amt', 0), errors='coerce')
            pchs_amt_smtot_amt = pd.to_numeric(summary.get('pchs_amt_smtot_amt', 0), errors='coerce')
            evlu_pfls_smtot_amt = pd.to_numeric(summary.get('evlu_pfls_smtot_amt', 0), errors='coerce')
            
            # 총 수익률 계산 (매입금액 대비 평가손익)
            total_return_rate = (evlu_pfls_smtot_amt / pchs_amt_smtot_amt * 100) if pchs_amt_smtot_amt > 0 else 0.0
            
            print("\n" + "="*80)
            print("【 계좌 총괄 수익 현황 】")
            print("="*80)
            print(f"▶ 총 평가금액: {tot_evlu_amt:,.0f} 원")
            print(f"▶ 총 수익률  : {total_return_rate:+.2f} %")
            print(f"▶ 총 평가손익: {evlu_pfls_smtot_amt:+,.0f} 원 (매입합계: {pchs_amt_smtot_amt:,.0f} 원)")
            
    except Exception as e:
        logger.error(f"보유 종목 조회 실패: {str(e)}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n조회 완료!")


if __name__ == "__main__":
    main()
