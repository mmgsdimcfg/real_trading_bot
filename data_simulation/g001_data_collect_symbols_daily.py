# -*- coding: utf-8 -*-
"""
g001_data_collect_symbols_daily.py

Purpose:
- Collect minute bars for a target date from KIS APIs.
- Build normalized intraday outputs and indicator columns used by live/sim flows.

Output schema (_10s.txt):
- datetime, open, high, low, close, volume, market, amount, raw_bar
  (amount = per-minute traded value on raw_bar==1 rows only; raw_bar 1 = KIS minute bar, 0 = interpolated)
- R76 indicator columns (MA_5, VOL_MA20, BB_*, RSI*, STOCH*, WILLIAMS*, MACD*, DI/ADX, VWAP, OBV*) only with
  --save-10s-indicators (placed before amount/raw_bar)
Per date folder also: {code}_{name}_daily.csv, _prev_close.json, _daily_close.json, _52w_high_low.json,
  _quote_snapshot.json, _market_index.json, nxt_flags.json, _{date}_picks.txt, _g001_progress.jsonl

Key behavior:
- Collects by date from KIS minute API (`inquire_time_dailychartprice`).
- Uses market priority NX > J > UN for overlapping timestamps.
- Supports NXT pre/after sessions (08:00~19:59) when symbol is NXT-tradeable.
- Exports 10-second interpolated bars (`_10s.txt`) for r007 simulation input.
- Optionally saves legacy `_1m/_3m/_20s` files for backward compatibility (--save-legacy-files).

Usage examples:
- Single code on specific date (regular market only):
    python xgraph/auto_trading/g001_data_collect_symbols_daily.py --date 20260508 --code 067310
- Single code on specific date (include NXT market):
    python xgraph/auto_trading/g001_data_collect_symbols_daily.py --date 20260508 --code 067310 --nxt
- Multiple dates (space- or comma-separated):
    python xgraph/auto_trading/g001_data_collect_symbols_daily.py --date 20260508 20260511 20260512
    python xgraph/auto_trading/g001_data_collect_symbols_daily.py --date 20260508,20260511,20260512 --nxt
- Multiple codes on specific date:
    python xgraph/auto_trading/g001_data_collect_symbols_daily.py --date 20260508 --code 067310,005930 --nxt
- Full list from symbols file:
    python xgraph/auto_trading/g001_data_collect_symbols_daily.py --date 20260508 --symbols-file xgraph/auto_trading/g004_universe_symbols_master.txt

Update log format (append only):
- [YYYY-MM-DD] type=feat|fix|refactor|docs owner=<name>
    summary: <one line>
    impact: <collector/live/sim/common>
    compatibility: <backward-compatible|breaking>

Update log:
- [2026-10-10] type=feat owner=claude
    summary: P1 보류분 + P2.
      (1) 일봉 저장 20 -> 100행(fetch_and_save_daily_ohlcv lookback_days). API 1회 최대 100행(최신순)이라 호출 수 불변.
      g002의 MA60/RS(21봉)가 20행으로는 항상 공란이던 문제 해소 - g002 점수 입력(RSI/ATR/20일 수익률 등)도 더 긴
      이력으로 계산되므로 종목 선정 결과가 달라진다(전후 비교는 커밋 메시지 참조).
      (2) --flows: {code}_{name}_flows.csv에 투자자별(외국인/기관/개인) 순매수, 프로그램 순매수, 공매도, 신용 융자잔고
      일별 ~30거래일 저장(종목당 +4회 호출, 기본 꺼짐). 날짜 인자가 있어 과거 날짜 백필 가능. 이어받기 시 기존 종목은
      flows 파일만 추가로 받음. 신용은 약 2거래일 늦게 공표, 투자자 수급은 장 마감 직후 수집 시 잠정치일 수 있음.
    impact: collector
    compatibility: backward-compatible (daily.csv 행 수만 증가, --flows는 옵션)
- [2026-10-10] type=fix owner=claude
    summary: 데이터 수집 P1 (Codex 교차검토 권고).
      (1) 분봉 조회 실패를 "데이터 없음"과 구분: 벤더 래퍼는 오류도 빈 DataFrame으로 돌려줘 페이지 중간 실패 시
      하루 일부만 저장되거나 "empty"로 기록돼 이후 이어받기에서 영영 건너뛰었음. _fetch_minute_page가 직접
      호출해 rt_cd!=0이면 MinuteChartError -> 종목 단위 재시도/실패(진행기록 없음 -> 다음 실행에서 재수집).
      실측: 데이터 없음(다른 시장, 비NXT 종목의 NX/UN, 없는 코드)은 모두 rt_cd=0 + 0행.
      (2) 일봉 csv 실패 시 진행기록 daily=False -> 다음 실행에서 재수집.
      (3) 52주 고저: 조회 시점 값이라 과거 날짜 백필 시 미래가 섞임 - 최근 거래일(KOSPI 일별지수로 실행당 1회
      조회)이 대상일일 때만 사용, 아니면 버림(r002/g002는 로컬 일봉 계산으로 폴백). 발생일 비교만으로는 대상일
      당시 고가가 지금 52주 창에서 빠진 경우를 못 걸러 이 방식으로 변경(Codex 검토). 구버전 진행기록의 w52는
      대상일 당일 조회분만 복원. 결과가 비어도 _52w_high_low.json/_quote_snapshot.json을 덮어써 이전 파일 제거. 같은 inquire_price 응답의 기준가/상하한가/시장경고/단기과열/
      투자유의/정리매매/임시정지/VI/시총/회전율/PER·PBR/외국인·프로그램 순매수 등을 _quote_snapshot.json에
      저장(추가 호출 없음, point_in_time=최근 거래일==대상일일 때만 대상일 값으로 유효).
      (4) 분봉 acml_tr_pbmn(누적 거래대금)에서 분별 거래대금(amount) 계산해 10s 파일에 저장.
      (5) search_stock_info를 NXT 판정과 기본정보 캐시 갱신이 1회 응답 공유(종목당 최대 2회 -> 1회).
      (6) 기본정보 캐시 fetched_at을 대상일이 아닌 실제 조회일(KST)로 기록, 나이도 오늘 기준.
      (7) _market_index.json에 "dates" 키 추가(g002는 kospi/kosdaq 리스트만 읽음).
      (8) 10s 지표 열 기본 생략(--save-10s-indicators로 유지, 이어받기도 이 옵션을 반영), _3m.txt는
      --save-legacy-files로 이동 -
      둘 다 읽는 코드 없음(g003은 재계산). 10s 파일 약 450KB -> 150KB/종목.
    impact: collector
    compatibility: 10s 파일에서 지표 21열이 기본으로 빠지고 _3m.txt가 기본 생성되지 않음(소비자 없음 확인,
      필요 시 옵션으로 복원). 그 외 산출물은 키/열 추가만. 진행기록에 quote/daily 필드 추가(구버전 기록 호환).
- [2026-10-10] type=fix owner=claude
    summary: 데이터 정합성 P0 (Codex 교차검토 후 실데이터로 확인).
      (1) _prev_close.json: 날짜 폴더를 거슬러 처음 찾은 날의 종가를 쓰던 방식 -> 해당 날짜 {code}_daily.csv의
      직전 행(공식 종가, 진짜 직전 거래일). 20261008은 20261007을 나중에 수집해 2,275/2,463종목이 20261006
      종가를 전일종가로 갖고 있었음(000020: 5450, 정답 5610). daily csv 없는 종목만 폴더 폴백(직전 거래일
      폴더로 한정). (2) _daily_close.json: 분봉 15:30 봉 종가 -> daily csv 당일 행(공식 종가), 단 daily csv가
      대상일 다음날 이후 저장된 경우만(당일 저장분은 미확정 관측: 20261006 16:38 5420 vs 최종 5450) -
      20261008은 분봉 종가가 공식 종가와 2,098/2,463종목 불일치(000020 분봉 5360 vs 공식 5410, KIS 재조회 동일).
      (3) 10s 파일 마지막 열에 raw_bar(1=원본 분봉, 0=보간) 추가 - g003/g009가 :00 행으로 원본 분봉을 복원할 때
      무체결 분의 보간 :00 행(직전 분 거래량 복사)이 가짜 분봉으로 섞이던 문제(20261008 표본 308종목 중 168종목
      거래량 +5% 초과, 최대 4.6배). (4) interpolate_to_20sec: 깨진 한글 주석이 줄바꿈을 삼켜 df_indexed 대입이
      주석 처리돼 --save-legacy-files 시 NameError - 복구. (5) interpolate_to_10sec ±15% 클리핑 제거 -
      대체값이 ffill(같은 값)이라 아무것도 바꾸지 않던 코드(출력 불변).
    impact: collector/sim
    compatibility: backward-compatible (10s 열 1개 추가 - 이름 기반 리더만 존재, g003/g009는 열 없으면 기존
      동작; prev/daily close 값은 공식 종가로 바뀜. 기존 날짜 폴더 파일은 재수집/재생성 전까지 그대로)
- [2026-10-10] type=feat owner=claude
    summary: --date에 여러 날짜를 공백으로도 나열 가능(nargs="+"). 기존 콤마 구분과 혼용 가능,
      중복 날짜는 입력 순서를 유지한 채 제거.
    impact: collector
    compatibility: backward-compatible
- [2026-10-09] type=fix owner=claude
    summary: load_symbols가 g004 유니버스 파일의 "#" 줄을 건너뛰도록 read_csv(comment="#").
      기존에는 "# 046070" 같은 주석 줄 16개가 종목코드로 그대로 읽혀 API 조회 대상에 포함됐음.
      g010(위험종목 자동 주석)이 쓰는 "# code,name,market  # auto: 사유" 줄도 제외됨.
    impact: collector
    compatibility: backward-compatible
- [2026-10-09] type=feat owner=claude
    summary: 중단 후 재실행 시 이어받기(resume). 종목 처리 완료 시 {date}/_g001_progress.jsonl에
      1줄(status/nxt/w52/fetched_at) 기록 - 파일 저장 후 마지막에 기록하므로 처리 도중 중단된
      종목은 재수집됨. 재실행 시 기록된 종목은 다운로드를 건너뛰고 nxt/w52를 복원해 종료 시
      집계 파일(nxt_flags/_52w/_daily_close/_prev_close/picks)이 전체 종목을 유지. 진행파일이
      없는 기존 파일(이전 버전 실행분)은 10s/3m/daily.csv가 모두 있고 장마감(+10분, KST) 이후
      저장된 경우에만 채택하고 NXT 확인/52주 조회(종목당 1~2회 API)만 다시 수행. 장 마감 전
      저장분, --nxt 여부 변경, --save-legacy-files 추가 시에는 재수집. --no-resume으로 전체 재수집.
    impact: collector
    compatibility: backward-compatible (기본 동작이 이어받기로 변경; 산출물 형식 불변, 날짜 폴더에
      _g001_progress.jsonl 추가 - g002/g003의 *.txt/*.csv glob에는 걸리지 않음)
- [2026-10-09] type=fix owner=claude
    summary: KIS HTTP 연결 재사용 + 재시도. kis_auth가 호출마다 requests.get/post로 새 TCP+TLS
      연결을 맺어 Raspberry Pi 4에서 호출당 ~170ms(재사용 시 ~25ms)가 소요됨. main() 시작 시
      ka.requests를 Session 기반 프록시(_KisRequestsProxy)로 교체 - 한국투자 라이브러리 파일은
      수정하지 않음. EGW00201(초당 거래건수 초과)은 최대 5회 백오프 재시도(기존에는 빈 DF가
      반환돼 _parse_one_market 페이지 루프가 중단되며 해당 시간대가 조용히 누락됐음), GET의
      연결오류/타임아웃은 최대 3회 재시도, 요청 타임아웃 (5s, 20s) 신규 적용.
    impact: collector
    compatibility: backward-compatible (산출물 형식 불변)
- [2026-09-23] type=feat owner=claude
    summary: 사용자 요청("g002 종목선정 조건 재검토, 좋은 조건 누락 시 g001에 데이터 추가") +
      Codex 설계검토 - fetch_stock_basic_info()가 이미 매일 호출 중인 search_stock_info(CTPF1002R,
      관리종목/거래정지/업종 조회용)의 동일 응답에서 lstg_stqt(상장주수) 필드를 추가로 추출해
      캐시에 저장 (신규 API 호출 없음, 기존 응답의 미사용 필드였음 - KIS Open API SDK
      examples_llm/domestic_stock/search_stock_info/chk_search_stock_info.py로 필드명 확인).
      g002가 price*lstg_stqt로 시가총액을 계산해 정보성 컬럼으로만 노출(점수 미반영 - Codex
      권고: 상장주수는 자사주/전략적보유 포함이라 진짜 유동주식(float)의 부정확한 근사치이고,
      거래대금/ATR와 겹칠 수 있어 실제 예측력 검증 전엔 배점하지 않는 게 안전).
    impact: collector
    compatibility: backward-compatible (기존 admn_item_yn/tr_stop_yn/sector 필드는 그대로,
      lstg_stqt 필드가 캐시 JSON에 추가됨 - 없어도(구버전 캐시) None으로 폴백)
    추가 수정(같은 날): get_stock_basic_info_cached()의 30일 캐시 재사용 로직이 fetched_at
      나이만 보고 스키마 변경을 감지 못해, 실제 캐시 파일(_stock_basic_info_cache.json)의
      2,557종목 중 다수(2026-09-14 조회, 9일 전)가 lstg_stqty 필드 추가 이전에 저장된 채로
      "신선함" 판정을 받아 최대 30일간 lstg_stqty 없이 재사용될 뻔한 문제 발견+수정 -
      "lstg_stqty" 키 자체가 없는(구버전) 항목은 나이 무관 1회 강제 재조회하도록 변경.
- [2026-08-29] type=feat owner=copilot
    summary: g002 스캐너 검토에서 나온 3개 보류 항목(관리종목/거래정지 배제, 업종 분산,
      시장 레짐/RS)을 위한 KIS Open API 스냅샷 수집 추가. probe_nxt_tradeable()이 이미
      실사용 중인 dsf.search_stock_info(CTPF1002R, 주식기본조회) 호출 패턴을 그대로 재사용.
      (1) fetch_stock_basic_info/get_stock_basic_info_cached - 종목별 admn_item_yn(관리종목),
      tr_stop_yn(거래정지), idx_bztp_lcls/mcls/scls_cd_name(업종 대/중/소분류)를 조회해
      data_root/_stock_basic_info_cache.json(날짜 폴더가 아닌 최상위, 코드별 fetched_at
      포함)에 캐시 - 느리게 바뀌는 참조데이터라 BASIC_INFO_CACHE_MAX_AGE_DAYS(30일) 이내면
      재조회하지 않음.
      (2) fetch_market_index_daily - dsf.inquire_index_daily_price(국내업종 일자별지수,
      FHPUP02120000)로 KOSPI(0001)/KOSDAQ(1001) 최근 30거래일 종가를 날짜당 1회(종목당 아님)
      조회해 output_dir/_market_index.json으로 저장. g002의 RS/시장레짐 계산이 지금까지
      pykrx(KRX 직접 스크래핑, 이 저장소 밖 환경에서 네트워크 오류로 실패 확인됨)에만 의존하던
      것을 이미 인증된 KIS 세션으로 대체하기 위함 - g002는 이 스냅샷을 우선 사용하고 없으면
      기존 pykrx로 폴백(하위호환).
      *** 미검증: 이번 세션 환경은 KIS Open API 자격증명이 설정돼 있지 않아(kis_devlp.yaml이
      플레이스홀더 상태) 실제 API 응답으로 검증하지 못했음. 문법 검증과 fetch_52w_high_low/
      probe_nxt_tradeable의 기존 검증된 패턴을 그대로 재사용했다는 점 외의 보증은 없음 - 실서버
      최초 실행 시 로그(saved stock basic-info cache / saved market index snapshot 여부와
      건수)를 확인 필요. 실패해도 항상 빈 값/None 반환으로 폴백하도록 방어했으므로 기존
      수집 파이프라인 자체가 깨지지는 않음. ***
    impact: collector
    compatibility: backward-compatible (신규 API 실패 시 조용히 폴백, 기존 산출물 형식/내용 불변;
      단 종목당 API 호출이 최초 실행 시 1회, 이후 30일 주기로 추가됨)
- [2026-08-23] type=fix owner=copilot
    summary: prev_close 계산(_compute_prev_close_from_data 폴백 + 당일 _daily_close.json 생성)이
      "10s 파일의 마지막 행"을 그대로 썼는데, --nxt로 수집한 날은 마지막 행이 정규장 종가(15:30)가
      아니라 NXT 오후세션(~20:00) 체결가일 수 있어 r006 fetch_prev_close()(stck_sdpr/prdy_clpr 등
      공식 전일종가 필드)와 어긋날 수 있었음. 신규 _regular_session_last_close() 헬퍼로 두 지점 모두
      REGULAR_END(15:30, r003) 이하 행만 사용하도록 수정 - r007의 MAX_BUY_RISE_PCT_FROM_PREV_CLOSE
      게이트가 live와 동일한 기준가를 쓰도록 정합성 확보.
    impact: collector
    compatibility: backward-compatible (--nxt 미사용 날짜는 결과 동일, --nxt 사용 날짜만 prev_close
      값이 변경될 수 있음)
- [2026-07-21] type=feat owner=copilot
    summary: KIS inquire_price API로 실서버 52주 고가/저가(w52_hgpr/w52_lwpr) 스냅샷 추가 수집
      (fetch_52w_high_low), 종목별 누적하여 _52w_high_low.json으로 저장. r002가 이 값을 직접 읽어
      high_52w_ratio 계산에 쓰면 로컬 일봉(약 13~20일치) 최고값으로 진짜 52주 고가를 대체하던
      왜곡을 제거함. 종목당 API 1회 추가 호출(기존 일봉 fetch와 동일한 루프/env_dv/rate-limit 패턴 재사용).
    impact: collector
    compatibility: backward-compatible (API 실패 시 None 반환, r002는 기존 로컬 계산으로 폴백)
- [2026-06-28] type=fix owner=copilot
    summary: 일봉 lookback 10->20일, 10초봉 보간 이상값 클리핑(±15%초과), 3분봉 기본 저장
    impact: collector
    compatibility: backward-compatible
- [2026-06-25] type=feat owner=copilot
    summary: 일봉 데이터 취득 추가 (fetch_and_save_daily_ohlcv); inquire_daily_itemchartprice API로 이전 10 영업일 일봉 OHLCV 저장 ({code}_daily.csv) - r002 우하향 종목 필터에 사용
    impact: collector
    compatibility: backward-compatible
- [2026-05-10] type=docs owner=copilot
    summary: added standardized file header and expandable update-log format.
    impact: collector
    compatibility: backward-compatible
"""

from __future__ import annotations

import argparse
import os
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "live_trading"))  # r001_define_config, r002_strategy_core_shared
PROJECT_ROOT = Path(os.environ.get("OPEN_TRADING_API_ROOT", str(Path.home() / "git" / "open-trading-api")))
sys.path.insert(0, str(PROJECT_ROOT / "examples_llm"))
sys.path.insert(0, str(PROJECT_ROOT / "examples_user" / "domestic_stock"))
sys.path.insert(0, str(PROJECT_ROOT / "examples_llm" / "domestic_stock" / "inquire_time_dailychartprice"))
sys.path.insert(0, str(PROJECT_ROOT / "examples_llm" / "domestic_stock" / "inquire_daily_itemchartprice"))
sys.path.insert(0, str(PROJECT_ROOT / "examples_llm" / "domestic_stock" / "inquire_price"))
for _flow_api in ("investor_trade_by_stock_daily", "program_trade_by_stock_daily", "daily_short_sale", "daily_credit_balance"):
    sys.path.insert(0, str(PROJECT_ROOT / "examples_llm" / "domestic_stock" / _flow_api))

import kis_auth as ka
import domestic_stock_functions as dsf
from r001_define_config import (
    ADX_PERIOD,
    BB_PERIOD,
    BB_STD_MULTIPLIER,
    MA_PERIOD,
    MACD_FAST,
    MACD_SIGNAL_PERIOD,
    MACD_SLOW,
    OBV_MA_PERIOD,
    REGULAR_END,
    RSI_PERIOD,
    RSI_SIGNAL_PERIOD,
    STOCH_D_PERIOD,
    STOCH_K_PERIOD,
    VOLUME_MA_PERIOD,
    WILLIAMS_D_PERIOD,
    WILLIAMS_R_PERIOD,
)


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


# KIS HTTP wrapper (connection reuse + retry).
# kis_auth calls the bare requests.get/post per API call, which opens a new TCP+TLS
# connection every time (~170ms per call on Raspberry Pi 4 vs ~25ms when reused).
# Swapping kis_auth's module-level `requests` keeps the vendor library untouched.
KIS_HTTP_TIMEOUT = (5, 20)          # (connect, read) seconds; kis_auth sets none
KIS_RATE_LIMIT_RETRIES = 5          # EGW00201 "초당 거래건수 초과"
KIS_RATE_LIMIT_BACKOFF = 0.5        # seconds, multiplied by attempt number
KIS_NETWORK_RETRIES = 3             # GET only: connection error / timeout
KIS_NETWORK_BACKOFF = 1.0


class _KisRequestsProxy:
    """Drop-in for the `requests` module inside kis_auth, backed by one Session."""

    def __init__(self, real_requests) -> None:
        self._real = real_requests
        self._session = real_requests.Session()

    def __getattr__(self, name):
        # exceptions, Response, etc. still resolve to the real module.
        return getattr(self._real, name)

    def get(self, url, **kwargs):
        return self._request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._request("POST", url, **kwargs)

    def _request(self, method: str, url: str, **kwargs):
        kwargs.setdefault("timeout", KIS_HTTP_TIMEOUT)
        rate_attempt = 0
        net_attempt = 0
        while True:
            try:
                res = self._session.request(method, url, **kwargs)
            except (self._real.exceptions.ConnectionError, self._real.exceptions.Timeout) as exc:
                # POST is not retried: the request may already have been processed.
                if method != "GET" or net_attempt >= KIS_NETWORK_RETRIES:
                    raise
                net_attempt += 1
                logger.warning("KIS network error, retry %d/%d: %s", net_attempt, KIS_NETWORK_RETRIES, exc)
                time.sleep(KIS_NETWORK_BACKOFF * net_attempt)
                continue

            # Rate-limit rejections are refused before processing, so retrying is safe for both methods.
            if res.status_code != 200 and "EGW00201" in res.text and rate_attempt < KIS_RATE_LIMIT_RETRIES:
                rate_attempt += 1
                logger.info("KIS rate limit (EGW00201), retry %d/%d", rate_attempt, KIS_RATE_LIMIT_RETRIES)
                time.sleep(KIS_RATE_LIMIT_BACKOFF * rate_attempt)
                continue
            return res


def _install_kis_http_session() -> None:
    if not isinstance(ka.requests, _KisRequestsProxy):
        ka.requests = _KisRequestsProxy(ka.requests)


# Indicator parameters are imported from r001_define_config so collector, live,
# and simulation stay in sync when strategy settings change.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Date-based KIS minute collector (20260401-compatible schema)")
    parser.add_argument("--env", type=str, default="real", choices=["real", "demo"], help="API environment")
    parser.add_argument("--date", type=str, nargs="+", default=None, help="Target date(s) YYYYMMDD. Space- or comma-separated list (e.g. 20260508 20260511 or 20260508,20260511)")
    parser.add_argument("--code", type=str, default="", help="Collect only this code (6-digit). Comma-separated supported")
    parser.add_argument("--symbols-file", type=str, default=str(SCRIPT_DIR / "g004_universe_symbols_master.txt"), help="Path to g004_universe_symbols_master.txt")
    parser.add_argument("--watchlist-file", type=str, default="", help="r004-style watchlist file(s) with code,name per line. Comma-separated multiple paths.")
    parser.add_argument("--watchlist-only", action="store_true", help="Use --watchlist-file as the only symbol source, ignoring --symbols-file.")
    parser.add_argument("--data-root", type=str, default=str(SCRIPT_DIR / "data"), help="Output data root")
    parser.add_argument("--sleep", type=float, default=0.12, help="Sleep seconds between symbols")
    parser.add_argument("--nxt", action="store_true", help="Include NXT market data (08:00~20:00)")
    parser.add_argument(
        "--save-legacy-files",
        action="store_true",
        help="Also save legacy _1m/_3m/_20s files in addition to default _10s output",
    )
    parser.add_argument(
        "--flows",
        action="store_true",
        help="Also save {code}_{name}_flows.csv (investor/program/short-sale/credit daily series, +4 API calls per symbol)",
    )
    parser.add_argument(
        "--save-10s-indicators",
        action="store_true",
        help="Keep the R76 indicator columns in _10s files (not read by g003/g009; g003 recomputes them on 1m/3m bars)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore already-collected symbols for the date and re-download everything",
    )
    return parser.parse_args()


# Resume support: each finished symbol appends one JSON line to {date_dir}/_g001_progress.jsonl
# (written only after its files are saved). A restarted run skips those symbols and restores
# their nxt/w52 values, so the end-of-run aggregates (nxt_flags/_52w/_daily_close/picks)
# still cover every symbol, not just the ones fetched in the latest run.
RESUME_PROGRESS_FILE = "_g001_progress.jsonl"
RESUME_CLOSE_MARGIN_MIN = 10  # files written before close+margin may hold a partial day


def _symbol_output_paths(output_dir: Path, code: str, name: str, save_legacy: bool) -> list[Path]:
    safe_name = str(name).replace("/", "_").replace("\\", "_")
    paths = [output_dir / f"{code}_{safe_name}_10s.txt"]
    if save_legacy:
        paths += [
            output_dir / f"{code}_{safe_name}_3m.txt",
            output_dir / f"{code}_{safe_name}_1m.txt",
            output_dir / f"{code}_{safe_name}_20s.txt",
        ]
    return paths


def _collection_cutoff_ts(target_date: str, include_nxt: bool) -> float:
    """Epoch seconds after which data for target_date is final (KST, independent of host TZ)."""
    close_hhmm = "2000" if include_nxt else "1530"
    close_dt = datetime.strptime(target_date + close_hhmm, "%Y%m%d%H%M").replace(tzinfo=ZoneInfo("Asia/Seoul"))
    return (close_dt + timedelta(minutes=RESUME_CLOSE_MARGIN_MIN)).timestamp()


def _load_resume_progress(output_dir: Path) -> dict[str, dict]:
    path = output_dir / RESUME_PROGRESS_FILE
    records: dict[str, dict] = {}
    if not path.is_file():
        return records
    with open(path, encoding="utf-8") as _f:
        for line in _f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated last line from an interrupted write
            if isinstance(rec, dict) and rec.get("code"):
                records[str(rec["code"])] = rec  # later lines win
    return records


def _append_resume_progress(output_dir: Path, record: dict) -> None:
    with open(output_dir / RESUME_PROGRESS_FILE, "a", encoding="utf-8") as _f:
        _f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _resume_record_valid(
    rec: dict, paths: list[Path], include_nxt: bool, save_legacy: bool, cutoff_ts: float, save_ind: bool = False,
) -> bool:
    if rec.get("status") not in ("saved", "empty"):
        return False
    # Records before 2026-10-10 have no "ind10s" key; their 10s files always carried the indicators.
    if save_ind and not rec.get("ind10s", True):
        return False
    if bool(rec.get("include_nxt")) != include_nxt:
        return False
    if save_legacy and not rec.get("legacy"):
        return False
    if float(rec.get("fetched_at") or 0) < cutoff_ts:
        return False
    if rec["status"] == "saved":
        if rec.get("daily") is False:
            return False  # daily csv fetch failed - refetch so prev/daily close and g002 inputs exist
        return all(p.is_file() for p in paths)
    return True


def _last_row_time(csv_path: Path) -> str | None:
    """HH:MM:SS of the last data row of a 10s file (reads only the file tail)."""
    try:
        with open(csv_path, "rb") as _f:
            _f.seek(0, os.SEEK_END)
            _f.seek(max(0, _f.tell() - 4096))
            lines = [ln for ln in _f.read().decode("utf-8", errors="ignore").splitlines() if ln.strip()]
        return lines[-1].split(",", 1)[0].strip()[-8:] if lines else None
    except OSError:
        return None


def _record_w52_point_in_time(rec: dict, target_date: str) -> bool:
    """Whether a resume record's w52 belongs to target_date (see fetch_quote_snapshot)."""
    quote = rec.get("quote")
    if isinstance(quote, dict) and "point_in_time" in quote:
        return bool(quote["point_in_time"])
    # Records before 2026-10-10 stored the raw snapshot. Valid if target_date is still the latest session
    # (resume records are always fetched after its close), or if it was fetched on target_date itself.
    if latest_session_date() == target_date:
        return True
    try:
        fetched = datetime.fromtimestamp(float(rec.get("fetched_at") or 0), ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")
    except (TypeError, ValueError, OSError):
        return False
    return fetched == target_date


def _quote_record_fields(code: str, env_dv: str, target_date: str) -> dict:
    snap = fetch_quote_snapshot(code=code, env_dv=env_dv, target_date=target_date) or {}
    return {"w52": snap.get("w52"), "quote": snap.get("quote")}


def _adopt_existing_symbol_files(
    output_dir: Path, code: str, name: str, include_nxt: bool, save_legacy: bool, cutoff_ts: float, env_dv: str,
    target_date: str, save_ind: bool = False,
) -> dict | None:
    """Build a resume record for files saved by a run without a progress entry (e.g. pre-resume version).

    Requires every output file plus the daily csv (written last, so 10s/3m are complete) to exist
    and be newer than the market close. Re-queries only the cheap per-symbol metadata
    (NXT probe, 52w) that the aggregates need; the chart pages are not re-downloaded.
    """
    safe_name = str(name).replace("/", "_").replace("\\", "_")
    paths = _symbol_output_paths(output_dir, code, name, save_legacy)
    paths.append(output_dir / f"{code}_{safe_name}_daily.csv")
    try:
        if any(p.stat().st_mtime < cutoff_ts for p in paths):
            return None
    except OSError:
        return None  # a file is missing

    try:
        with open(paths[0], encoding="utf-8-sig") as _f:
            has_ind = "MA_5" in _f.readline().split(",")
    except OSError:
        return None
    if save_ind and not has_ind:
        return None  # indicators requested but missing - regenerate

    try:
        nxt = probe_nxt_tradeable(code) if include_nxt else False
    except Exception as exc:
        logger.debug("resume adopt: NXT probe failed for %s, refetching: %s", code, exc)
        return None
    if nxt:
        last_time = _last_row_time(paths[0])
        if last_time is None or last_time <= "15:30:00":
            return None  # collected without NXT session; refetch with --nxt data

    return {
        "code": code, "name": name, "status": "saved", "include_nxt": include_nxt,
        "legacy": save_legacy, "nxt": nxt, **_quote_record_fields(code, env_dv, target_date),
        "daily": True, "ind10s": has_ind, "fetched_at": time.time(), "adopted": True,
    }


def _is_truthy_flag(value) -> bool | None:
    if value in (None, ""):
        return None
    text = str(value).strip().upper()
    if text in {"Y", "1", "TRUE", "T", "O", "YES"}:
        return True
    if text in {"N", "0", "FALSE", "F", "X", "NO"}:
        return False
    return None


# search_stock_info (CTPF1002R) rows memoized per run: the NXT probe and the basic-info cache refresh
# read the same response, which used to be requested twice per symbol. Failures are not memoized.
_STOCK_INFO_ROWS: dict[str, pd.Series] = {}


def _search_stock_info_row(code: str) -> pd.Series | None:
    if code in _STOCK_INFO_ROWS:
        return _STOCK_INFO_ROWS[code]
    stock_info_fn = getattr(dsf, "search_stock_info", None)
    if not callable(stock_info_fn):
        return None

    last_exc = None
    result = None
    for attempt in range(2):
        try:
            result = stock_info_fn(prdt_type_cd="300", pdno=code)
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            if attempt == 0:
                logger.debug("search_stock_info attempt 1 failed for %s, retrying: %s", code, exc)
                time.sleep(0.3)

    if last_exc is not None:
        logger.warning("search_stock_info failed for %s (both attempts): %s", code, last_exc)
        return None
    if result is None or getattr(result, "empty", True):
        return None
    row = result.iloc[-1]
    _STOCK_INFO_ROWS[code] = row
    return row


def probe_nxt_tradeable(code: str) -> bool:
    row = _search_stock_info_row(code)
    if row is None:
        return False

    for key in ("cptt_trad_tr_psbl_yn", "nxt_tr_stop_yn", "tr_stop_yn"):
        if key not in row.index:
            continue
        flag = _is_truthy_flag(row.get(key))
        if flag is None:
            continue
        return flag if key == "cptt_trad_tr_psbl_yn" else (not flag)

    return False


MINUTE_CHART_URL = "/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice"
MINUTE_CHART_TR_ID = "FHKST03010230"  # 주식일별분봉조회


class MinuteChartError(RuntimeError):
    """KIS minute-chart request failed (not "no data")."""


def _fetch_minute_page(code: str, market_div: str, target_date: str, hour: str) -> pd.DataFrame:
    """One page of inquire_time_dailychartprice output2; raises MinuteChartError on an API error.

    The vendor wrapper returns an empty DataFrame for both "no data" and errors, so a failed page
    used to end pagination silently: the symbol was saved with a truncated day, or recorded as
    "empty" and skipped by every later resume. Measured: no-data cases (other market, NX/UN for a
    non-NXT symbol, unknown code) all answer rt_cd=0 with zero rows, so not-OK is always an error.
    """
    res = ka._url_fetch(MINUTE_CHART_URL, MINUTE_CHART_TR_ID, "", {
        "FID_COND_MRKT_DIV_CODE": market_div,
        "FID_INPUT_ISCD": code,
        "FID_INPUT_HOUR_1": hour,
        "FID_INPUT_DATE_1": target_date,
        "FID_PW_DATA_INCU_YN": "Y",
        "FID_FAKE_TICK_INCU_YN": "",
    })
    if not res.isOK():
        try:
            detail = f"{res.getErrorCode()} {res.getErrorMessage()}"
        except Exception:
            detail = str(getattr(res, "getResCode", lambda: "?")())
        raise MinuteChartError(f"{code} {market_div} {target_date} {hour}: {detail}")
    return pd.DataFrame(res.getBody().output2)


def _parse_one_market(code: str, market_div: str, target_date: str, max_pages: int = 15) -> pd.DataFrame | None:
    # Regular market: until 15:30, NXT: until 20:00.
    # API errors propagate (MinuteChartError) so the caller retries/fails the whole symbol
    # instead of saving a partial day.
    current_hour = "200000" if market_div == "NX" else "153000"
    all_df: list[pd.DataFrame] = []
    last_earliest_time: int | None = None

    for _ in range(max_pages):
        df = _fetch_minute_page(code, market_div, target_date, current_hour)

        if df is None or df.empty:
            break

        if "stck_bsop_date" in df.columns:
            df = df[df["stck_bsop_date"].astype(str) == target_date].copy()
        if df.empty:
            break

        all_df.append(df)

        times = pd.to_numeric(df.get("stck_cntg_hour"), errors="coerce").dropna().astype("int64")
        if times.empty:
            break

        earliest = int(times.min())
        if last_earliest_time is not None and earliest >= last_earliest_time:
            break
        last_earliest_time = earliest

        if earliest <= 80000:
            break

        next_dt = datetime.strptime(f"{earliest:06d}", "%H%M%S") - timedelta(minutes=1)
        current_hour = next_dt.strftime("%H%M%S")
        time.sleep(0.1)

    if not all_df:
        return None

    merged = pd.concat(all_df, ignore_index=True)
    merged["stck_cntg_hour"] = merged["stck_cntg_hour"].astype(str).str.zfill(6)

    for col in ("stck_oprc", "stck_hgpr", "stck_lwpr", "stck_prpr", "cntg_vol"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    if "acml_tr_pbmn" in merged.columns:
        merged["acml_tr_pbmn"] = pd.to_numeric(merged["acml_tr_pbmn"], errors="coerce")
    else:
        merged["acml_tr_pbmn"] = float("nan")

    merged["datetime"] = pd.to_datetime(
        target_date + merged["stck_cntg_hour"],
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )
    merged = merged.dropna(subset=["datetime"]).copy()

    merged["_has_trade"] = (merged["cntg_vol"].fillna(0) > 0).astype("int64")
    merged = merged.sort_values(
        by=["datetime", "_has_trade", "cntg_vol"],
        ascending=[True, False, False],
    ).drop_duplicates(subset=["datetime"], keep="first")

    # Per-minute traded value (KRW) = diff of this market's cumulative value (acml_tr_pbmn);
    # the first bar of the day is its own cumulative value. Taken before the low-price filter so
    # dropping a bad row does not merge two minutes' value into one.
    merged = merged.sort_values("datetime")
    acml = merged["acml_tr_pbmn"]
    amount = acml.diff()
    amount.iloc[0] = acml.iloc[0]
    merged["amount"] = amount.where(amount >= 0)

    valid_price = merged[["stck_oprc", "stck_hgpr", "stck_lwpr", "stck_prpr"]].max(axis=1) > 0
    merged = merged[valid_price].copy()
    if merged.empty:
        return None

    out = merged.rename(
        columns={
            "stck_oprc": "open",
            "stck_hgpr": "high",
            "stck_lwpr": "low",
            "stck_prpr": "close",
            "cntg_vol": "volume",
        }
    )[["datetime", "open", "high", "low", "close", "volume", "amount"]]
    out["market"] = market_div
    return out.sort_values("datetime").reset_index(drop=True)


def fetch_symbol_data(code: str, target_date: str, include_nxt: bool) -> pd.DataFrame | None:
    market_candidates = ["J", "UN"]
    if include_nxt:
        market_candidates = ["NX", "J", "UN"]

    collected: list[pd.DataFrame] = []
    for market in market_candidates:
        data = _parse_one_market(code, market, target_date)
        if data is not None and not data.empty:
            collected.append(data)

    if not collected:
        return None

    merged = pd.concat(collected, ignore_index=True)

    # Filter data based on market hours
    if not include_nxt:
        merged = merged[merged["datetime"].dt.time >= datetime.strptime("09:00:00", "%H:%M:%S").time()]

    # One row per minute. Priority: NX > J > UN.
    priority = {"NX": 0, "J": 1, "UN": 2}
    merged["_priority"] = merged["market"].map(priority).fillna(9).astype(int)
    merged = merged.sort_values(["datetime", "_priority"]).drop_duplicates(subset=["datetime"], keep="first")
    merged = merged.drop(columns=["_priority"]).sort_values("datetime").reset_index(drop=True)

    return merged


def calculate_r76_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    out["MA_5"] = out["close"].rolling(window=MA_PERIOD, min_periods=1).mean()
    out["VOL_MA20"] = out["volume"].rolling(window=VOLUME_MA_PERIOD, min_periods=1).mean()

    out["BB_MIDDLE"] = out["close"].rolling(window=BB_PERIOD, min_periods=1).mean()
    out["BB_STD"] = out["close"].rolling(window=BB_PERIOD, min_periods=1).std()
    out["BB_UPPER"] = out["BB_MIDDLE"] + out["BB_STD"] * BB_STD_MULTIPLIER
    out["BB_LOWER"] = out["BB_MIDDLE"] - out["BB_STD"] * BB_STD_MULTIPLIER

    delta = out["close"].diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1.0 / RSI_PERIOD, min_periods=1, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / RSI_PERIOD, min_periods=1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    out["RSI"] = 100 - (100 / (1 + rs))
    out.loc[avg_loss == 0, "RSI"] = 100.0
    out["RSI_SIGNAL"] = out["RSI"].rolling(window=RSI_SIGNAL_PERIOD, min_periods=1).mean()

    low_n = out["low"].rolling(window=STOCH_K_PERIOD, min_periods=1).min()
    high_n = out["high"].rolling(window=STOCH_K_PERIOD, min_periods=1).max()
    denom = (high_n - low_n).replace(0, float("nan"))
    out["STOCH_K"] = 100.0 * (out["close"] - low_n) / denom
    out["STOCH_D"] = out["STOCH_K"].rolling(window=STOCH_D_PERIOD, min_periods=1).mean()

    high_w = out["high"].rolling(window=WILLIAMS_R_PERIOD, min_periods=1).max()
    low_w = out["low"].rolling(window=WILLIAMS_R_PERIOD, min_periods=1).min()
    wr_denom = (high_w - low_w).replace(0, float("nan"))
    out["WILLIAMS_R"] = -100.0 * (high_w - out["close"]) / wr_denom
    out["WILLIAMS_D"] = out["WILLIAMS_R"].rolling(window=WILLIAMS_D_PERIOD, min_periods=1).mean()

    ema_fast = out["close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = out["close"].ewm(span=MACD_SLOW, adjust=False).mean()
    out["MACD"] = ema_fast - ema_slow
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]

    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - out["close"].shift(1)).abs(),
            (out["low"] - out["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    high_diff = out["high"] - out["high"].shift(1)
    low_diff = out["low"].shift(1) - out["low"]
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)
    ema_tr = tr.ewm(alpha=1.0 / ADX_PERIOD, min_periods=1, adjust=False).mean()
    ema_plus = plus_dm.ewm(alpha=1.0 / ADX_PERIOD, min_periods=1, adjust=False).mean()
    ema_minus = minus_dm.ewm(alpha=1.0 / ADX_PERIOD, min_periods=1, adjust=False).mean()
    out["DI_PLUS"] = 100.0 * ema_plus / ema_tr.replace(0, float("nan"))
    out["DI_MINUS"] = 100.0 * ema_minus / ema_tr.replace(0, float("nan"))
    di_sum = (out["DI_PLUS"] + out["DI_MINUS"]).replace(0, float("nan"))
    dx = 100.0 * (out["DI_PLUS"] - out["DI_MINUS"]).abs() / di_sum
    out["ADX"] = dx.ewm(alpha=1.0 / ADX_PERIOD, min_periods=1, adjust=False).mean()

    cum_vol = out["volume"].cumsum()
    out["VWAP"] = (out["close"] * out["volume"]).cumsum() / cum_vol.replace(0, float("nan"))

    close_diff = out["close"].diff()
    obv_vol = out["volume"] * close_diff.gt(0).astype(float) - out["volume"] * close_diff.lt(0).astype(float)
    out["OBV"] = obv_vol.cumsum()
    out["OBV_MA"] = out["OBV"].rolling(window=OBV_MA_PERIOD, min_periods=1).mean()

    return out


def interpolate_to_20sec(minute_df: pd.DataFrame) -> pd.DataFrame:
    """1遺꾨큺 ?곗씠?곕? 20珥?媛꾧꺽?쇰줈 ?좏삎 蹂닿컙???뺤옣?쒕떎.

    Note:
    - ???곗씠?곕뒗 ?덇굅???명솚 紐⑹쟻??蹂닿컙 寃곌낵?대ŉ,
      ?쒕쾭?먯꽌 20珥?二쇨린濡?吏곸젒 ?섏쭛??泥닿껐/?멸? ?곗씠?곌? ?꾨땲??
    """
    df = minute_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    
    if df.empty:
        return df
    
    # datetime을 인덱스로 지정
    df_indexed = df.set_index("datetime")
    
    # 20珥?媛꾧꺽???쒓컙 ?앹꽦
    time_range_20sec = pd.date_range(
        start=df_indexed.index.min(),
        end=df_indexed.index.max(),
        freq="20s"
    )
    
    # datetime ?몃뜳?ㅻ줈 由ъ씤?깆떛
    df_reindexed = df_indexed.reindex(time_range_20sec.union(df_indexed.index)).sort_index()
    
    # 媛寃??곗씠???좏삎 蹂닿컙
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        df_reindexed[col] = df_reindexed[col].interpolate(method="linear", limit_direction="both")
    
    # 嫄곕옒?됱? 20珥?遊??⑷퀎媛 ?먮낯 遺꾨큺 ?⑷퀎? 理쒕????쇱튂?섎룄濡?遺꾪븷?쒕떎.
    step_seconds = 20.0
    if len(df.index) >= 2:
        src_seconds = df["datetime"].diff().dropna().dt.total_seconds().median()
        if pd.notna(src_seconds) and src_seconds > 0:
            expansion = max(1.0, round(float(src_seconds) / step_seconds))
        else:
            expansion = 3.0
    else:
        expansion = 3.0

    df_reindexed["volume"] = df_reindexed["volume"].ffill()
    df_reindexed["volume"] = pd.to_numeric(df_reindexed["volume"], errors="coerce") / expansion
    df_reindexed["volume"] = df_reindexed["volume"].fillna(0)
    
    # Forward-fill market column if present.
    if "market" in df_reindexed.columns:
        df_reindexed["market"] = df_reindexed["market"].ffill()
    
    # 吏??而щ읆? ?댁쟾 媛믪쑝濡?梨꾩?(1遊??댁긽?대㈃ 洹몃?濡?蹂듭궗)
    indicator_cols = [
        "MA_5", "VOL_MA20", "BB_MIDDLE", "BB_STD", "BB_UPPER", "BB_LOWER",
        "RSI", "RSI_SIGNAL", "STOCH_K", "STOCH_D", "WILLIAMS_R", "WILLIAMS_D",
        "MACD", "MACD_SIGNAL", "MACD_HIST", "DI_PLUS", "DI_MINUS", "ADX",
        "VWAP", "OBV", "OBV_MA",
    ]
    for col in indicator_cols:
        if col in df_reindexed.columns:
            df_reindexed[col] = df_reindexed[col].ffill()
    
    # Keep only 20-second rows.
    df_20sec = df_reindexed.loc[time_range_20sec].reset_index()
    df_20sec.rename(columns={"index": "datetime"}, inplace=True)
    
    return df_20sec


def interpolate_to_10sec(minute_df: pd.DataFrame) -> pd.DataFrame:
    """1遺꾨큺 ?곗씠?곕? 10珥?媛꾧꺽?쇰줈 ?좏삎 蹂닿컙???뺤옣?쒕떎.

    Note:
    - ???곗씠?곕뒗 ?덇굅???명솚/?쒕??덉씠???뺣????μ긽 紐⑹쟻??蹂닿컙 寃곌낵?대ŉ,
      ?쒕쾭?먯꽌 10珥?二쇨린濡?吏곸젒 ?섏쭛??泥닿껐 ?곗씠?곌? ?꾨땲??
    """
    df = minute_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    if df.empty:
        return df

    df_indexed = df.set_index("datetime")
    time_range_10sec = pd.date_range(
        start=df_indexed.index.min(),
        end=df_indexed.index.max(),
        freq="10s"
    )

    df_reindexed = df_indexed.reindex(time_range_10sec.union(df_indexed.index)).sort_index()

    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        df_reindexed[col] = df_reindexed[col].interpolate(method="linear", limit_direction="both")

    step_seconds = 10.0
    if len(df.index) >= 2:
        src_seconds = df["datetime"].diff().dropna().dt.total_seconds().median()
        if pd.notna(src_seconds) and src_seconds > 0:
            expansion = max(1.0, round(float(src_seconds) / step_seconds))
        else:
            expansion = 6.0
    else:
        expansion = 6.0

    df_reindexed["volume"] = df_reindexed["volume"].ffill()
    df_reindexed["volume"] = pd.to_numeric(df_reindexed["volume"], errors="coerce") / expansion
    df_reindexed["volume"] = df_reindexed["volume"].fillna(0)

    if "market" in df_reindexed.columns:
        df_reindexed["market"] = df_reindexed["market"].ffill()

    indicator_cols = [
        "MA_5", "VOL_MA20", "BB_MIDDLE", "BB_STD", "BB_UPPER", "BB_LOWER",
        "RSI", "RSI_SIGNAL", "STOCH_K", "STOCH_D", "WILLIAMS_R", "WILLIAMS_D",
        "MACD", "MACD_SIGNAL", "MACD_HIST", "DI_PLUS", "DI_MINUS", "ADX",
        "VWAP", "OBV", "OBV_MA",
    ]
    for col in indicator_cols:
        if col in df_reindexed.columns:
            df_reindexed[col] = df_reindexed[col].ffill()

    df_10sec = df_reindexed.loc[time_range_10sec].reset_index()
    df_10sec.rename(columns={"index": "datetime"}, inplace=True)
    return df_10sec


def build_3min_indicator_frame(minute_df: pd.DataFrame) -> pd.DataFrame:
    base = minute_df.copy()
    base["datetime"] = pd.to_datetime(base["datetime"], errors="coerce")
    base = base.dropna(subset=["datetime"]).sort_values("datetime")
    if base.empty:
        return pd.DataFrame(columns=base.columns)

    idx = base.set_index("datetime")
    bars_3m = idx.resample("3min", label="right", closed="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    bars_3m = bars_3m.dropna(subset=["open", "high", "low", "close"])
    if bars_3m.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    bars_3m = calculate_r76_indicators(bars_3m)
    return bars_3m.reset_index().rename(columns={"index": "datetime"})


def enrich_with_strategy_indicators(minute_df: pd.DataFrame) -> pd.DataFrame:
    base = minute_df.copy()
    base["datetime"] = pd.to_datetime(base["datetime"], errors="coerce")
    base = base.dropna(subset=["datetime"]).sort_values("datetime")
    if base.empty:
        return base

    bars_3m = build_3min_indicator_frame(base)
    if bars_3m.empty:
        return base

    indicator_cols = [
        "MA_5", "VOL_MA20", "BB_MIDDLE", "BB_STD", "BB_UPPER", "BB_LOWER",
        "RSI", "RSI_SIGNAL", "STOCH_K", "STOCH_D", "WILLIAMS_R", "WILLIAMS_D",
        "MACD", "MACD_SIGNAL", "MACD_HIST", "DI_PLUS", "DI_MINUS", "ADX",
        "VWAP", "OBV", "OBV_MA",
    ]
    bars_for_merge = bars_3m[["datetime", *indicator_cols]].copy()

    enriched = pd.merge_asof(
        base.sort_values("datetime"),
        bars_for_merge.sort_values("datetime"),
        on="datetime",
        direction="backward",
    )

    return enriched


def fetch_and_save_daily_ohlcv(
    code: str,
    name: str,
    env_dv: str,
    target_date: str,
    output_dir: Path,
    lookback_days: int = 100,
) -> bool:
    """Fetch actual daily (일봉) OHLCV for the past lookback_days business days.
    Saves {code}_{name}_daily.csv to output_dir for r002 downtrend filter.
    Uses KIS inquire_daily_itemchartprice API.  Returns True on success.

    [2026-10-10] 20 -> 100 rows: one call returns at most 100 rows (newest first), so this is the
    most a single call gives - MA60/RS(21 bars) in g002 were always blank with 20 rows. No extra calls.
    """
    try:
        from inquire_daily_itemchartprice import inquire_daily_itemchartprice as _daily_api
    except ImportError:
        logger.debug("inquire_daily_itemchartprice not importable; skipping daily OHLCV for %s", code)
        return False

    target_dt = datetime.strptime(target_date, "%Y%m%d")
    date_from = (target_dt - timedelta(days=lookback_days * 2 + 5)).strftime("%Y%m%d")

    try:
        _, df2 = _daily_api(
            env_dv=env_dv,
            fid_cond_mrkt_div_code="J",
            fid_input_iscd=code,
            fid_input_date_1=date_from,
            fid_input_date_2=target_date,
            fid_period_div_code="D",
            fid_org_adj_prc="0",
        )
    except Exception as exc:
        logger.debug("daily OHLCV fetch failed %s: %s", code, exc)
        return False

    if df2 is None or df2.empty:
        return False

    col_map = {
        "stck_bsop_date": "date",
        "stck_oprc": "open",
        "stck_hgpr": "high",
        "stck_lwpr": "low",
        "stck_clpr": "close",
        "acml_vol": "volume",
        "acml_tr_pbmn": "amount",
    }
    df2 = df2.rename(columns={k: v for k, v in col_map.items() if k in df2.columns})
    keep = [c for c in ("date", "open", "high", "low", "close", "volume", "amount") if c in df2.columns]
    if "date" not in keep or "close" not in keep:
        return False

    df2 = df2[keep].copy()
    for col in ("open", "high", "low", "close", "volume", "amount"):
        if col in df2.columns:
            df2[col] = pd.to_numeric(df2[col], errors="coerce")
    df2 = df2.dropna(subset=["close"]).sort_values("date").tail(lookback_days)
    if df2.empty:
        return False

    safe_name = str(name).replace("/", "_").replace("\\", "_")
    out_path = output_dir / f"{code}_{safe_name}_daily.csv"
    df2.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.debug("[daily] %s(%s) | %d rows -> %s", code, name, len(df2), out_path)
    return True


# --flows: per-symbol daily supply/demand series -> {code}_{name}_flows.csv (one row per date, ~30 trading
# days up to target_date; each API answers one call with ~30 rows and takes a date, so past dates backfill).
# Columns keep the KIS field names, prefixed by source. Availability timing differs - consumers must lag:
#   inv_*  투자자별 순매수(외국인/기관/개인, 수량·대금 - 대금 단위 백만원): KRX finalizes after the close; a run
#          right after 15:30 can see preliminary values for target_date.
#   pgm_*  프로그램매매 순매수(수량·대금 - 대금 단위 원).
#   ss_*   공매도 체결수량/거래량 대비 비중(%)/대금(원).
#   crd_*  신용 융자잔고(주수/잔고율), indexed by deal_date: published ~2 trading days late (20261008 run -> last 20261006).
FLOW_FIELDS = {
    "inv": ("frgn_ntby_qty", "orgn_ntby_qty", "prsn_ntby_qty", "frgn_ntby_tr_pbmn", "orgn_ntby_tr_pbmn", "prsn_ntby_tr_pbmn"),
    "pgm": ("whol_smtn_ntby_qty", "whol_smtn_ntby_tr_pbmn"),
    "ss": ("ssts_cntg_qty", "ssts_vol_rlim", "ssts_tr_pbmn"),
    "crd": ("whol_loan_rmnd_stcn", "whol_loan_rmnd_rate"),
}


def _flow_frame(df: pd.DataFrame | None, prefix: str, date_col: str = "stck_bsop_date") -> pd.DataFrame | None:
    if df is None or df.empty or date_col not in df.columns:
        return None
    cols = [c for c in FLOW_FIELDS[prefix] if c in df.columns]
    if not cols:
        return None
    out = df[[date_col, *cols]].copy()
    out[date_col] = out[date_col].astype(str).str.strip()
    out = out[out[date_col].str.fullmatch(r"\d{8}")]
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.rename(columns={date_col: "date", **{c: f"{prefix}_{c}" for c in cols}})
    return out.drop_duplicates("date").set_index("date")


FLOW_SOURCES = tuple(FLOW_FIELDS)  # ("inv", "pgm", "ss", "crd")
FLOW_MAX_ATTEMPTS = 3  # a source still empty after this many runs is treated as legitimately empty


def _fetch_flow_source(src: str, code: str, target_date: str) -> pd.DataFrame | None:
    try:
        if src == "inv":
            from investor_trade_by_stock_daily import investor_trade_by_stock_daily as _inv
            _o1, o2 = _inv(fid_cond_mrkt_div_code="J", fid_input_iscd=code, fid_input_date_1=target_date,
                           fid_org_adj_prc="", fid_etc_cls_code="", max_depth=1)
            return _flow_frame(o2, "inv")
        if src == "pgm":
            from program_trade_by_stock_daily import program_trade_by_stock_daily as _pgm
            return _flow_frame(_pgm(fid_cond_mrkt_div_code="J", fid_input_iscd=code, fid_input_date_1=target_date), "pgm")
        if src == "ss":
            from daily_short_sale import daily_short_sale as _ss
            date_from = (datetime.strptime(target_date, "%Y%m%d") - timedelta(days=45)).strftime("%Y%m%d")
            _o1, o2 = _ss(fid_cond_mrkt_div_code="J", fid_input_iscd=code, fid_input_date_1=date_from, fid_input_date_2=target_date)
            return _flow_frame(o2, "ss")
        if src == "crd":
            from daily_credit_balance import daily_credit_balance as _crd
            return _flow_frame(
                _crd(fid_cond_mrkt_div_code="J", fid_cond_scr_div_code="20476", fid_input_iscd=code,
                     fid_input_date_1=target_date, max_depth=1),
                "crd", date_col="deal_date",
            )
    except Exception as exc:
        logger.debug("flow source %s failed %s: %s", src, code, exc)
    return None


def fetch_and_save_flows(
    code: str, name: str, target_date: str, output_dir: Path, sources: tuple[str, ...] = FLOW_SOURCES,
) -> list[str]:
    """Fetch the given flow sources (1 call each) and merge them into {code}_{name}_flows.csv, keeping
    columns of sources already saved by an earlier run. Returns the sources still missing.

    The vendor wrappers return empty frames on errors as well as on "no data", so an empty source is
    reported missing and retried by later runs (up to FLOW_MAX_ATTEMPTS, see the resume branch in main).
    """
    safe_name = str(name).replace("/", "_").replace("\\", "_")
    path = output_dir / f"{code}_{safe_name}_flows.csv"
    existing = None
    if path.exists():
        try:
            existing = pd.read_csv(path, dtype={"date": str}).set_index("date")
        except Exception:
            existing = None

    fetched: dict[str, pd.DataFrame] = {}
    for src in sources:
        frame = _fetch_flow_source(src, code, target_date)
        if frame is not None:
            frame = frame[frame.index <= target_date]
        if frame is not None and not frame.empty:
            fetched[src] = frame

    parts: list[pd.DataFrame] = []
    if existing is not None:
        keep = [c for c in existing.columns if c.split("_", 1)[0] not in fetched]
        if keep:
            parts.append(existing[keep])
    parts.extend(fetched.values())
    present = {c.split("_", 1)[0] for part in parts for c in part.columns}
    if fetched:
        merged = pd.concat(parts, axis=1).sort_index()
        merged.index.name = "date"
        merged.reset_index().to_csv(path, index=False, encoding="utf-8-sig")
    return [src for src in FLOW_SOURCES if src not in present]


# inquire_price fields kept in _quote_snapshot.json (same response as the 52w lookup, no extra call).
# This is the state at fetch time: valid for target_date only while target_date is still the latest
# session (collected that evening / before the next open) - "point_in_time" in each entry.
QUOTE_NUMERIC_FIELDS = (
    "stck_sdpr", "stck_mxpr", "stck_llam", "aspr_unit", "hts_avls", "vol_tnrt", "per", "pbr", "eps", "bps",
    "hts_frgn_ehrt", "frgn_ntby_qty", "pgtr_ntby_qty", "marg_rate", "whol_loan_rmnd_rate",
)
QUOTE_FLAG_FIELDS = (
    "iscd_stat_cls_code", "mrkt_warn_cls_code", "short_over_yn", "invt_caful_yn", "sltr_yn", "temp_stop_yn",
    "mang_issu_cls_code", "vi_cls_code", "crdt_able_yn", "ssts_yn", "w52_hgpr_date", "w52_lwpr_date",
)


def _kst_today() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")


_LATEST_SESSION: dict[str, str | None] = {}


def latest_session_date() -> str | None:
    """Most recent KRX session date (YYYYMMDD), from the KOSPI daily index as of today (KST); once per run."""
    today = _kst_today()
    if today in _LATEST_SESSION:
        return _LATEST_SESSION[today]
    latest = None
    index_fn = getattr(dsf, "inquire_index_daily_price", None)
    if callable(index_fn):
        try:
            _df1, df2 = index_fn(
                fid_period_div_code="D", fid_cond_mrkt_div_code="U", fid_input_iscd="0001", fid_input_date_1=today,
            )
            if df2 is not None and not df2.empty and "stck_bsop_date" in df2.columns:
                dates = df2["stck_bsop_date"].astype(str).str.strip()
                dates = dates[dates.str.fullmatch(r"\d{8}") & (dates <= today)]
                latest = dates.max() if not dates.empty else None
        except Exception as exc:
            logger.warning("latest session lookup failed: %s", exc)
    if latest is None:
        logger.warning("latest session date unknown - quote/52w snapshots treated as point-in-time only when fetched on the target date")
    _LATEST_SESSION[today] = latest
    return latest


def fetch_quote_snapshot(code: str, env_dv: str, target_date: str) -> dict | None:
    """KIS inquire_price snapshot -> {"w52": {...} | None, "quote": {...}} or None on failure.

    w52 (w52_hgpr/w52_lwpr) feeds r002/g002 high_52w_ratio. The API has no date parameter and reflects
    every session up to now, so it is target_date's value only while target_date is still the latest
    session (point_in_time). Otherwise w52 is dropped and consumers fall back to their local daily-bar
    calc: checking the high/low dates alone is not enough, since target_date's 52-week high may have
    rolled out of today's window (Codex review).
    """
    try:
        from inquire_price import inquire_price as _price_api
    except ImportError:
        logger.debug("inquire_price not importable; skipping quote snapshot for %s", code)
        return None

    try:
        df = _price_api(env_dv=env_dv, fid_cond_mrkt_div_code="J", fid_input_iscd=code)
    except Exception as exc:
        logger.debug("quote snapshot fetch failed %s: %s", code, exc)
        return None

    if df is None or df.empty:
        return None

    row = df.iloc[0]
    quote: dict = {"fetched_on": _kst_today()}
    latest = latest_session_date()
    quote["point_in_time"] = (latest == target_date) if latest else (quote["fetched_on"] == target_date)
    for key in QUOTE_NUMERIC_FIELDS:
        try:
            quote[key] = float(row.get(key))
        except (TypeError, ValueError):
            quote[key] = None
    for key in QUOTE_FLAG_FIELDS:
        val = row.get(key)
        quote[key] = None if val is None else str(val).strip()

    w52 = None
    try:
        w52_high = float(row.get("w52_hgpr"))
        w52_low = float(row.get("w52_lwpr"))
    except (TypeError, ValueError):
        w52_high = w52_low = 0.0
    if w52_high > 0:
        if quote["point_in_time"]:
            w52 = {"w52_high": w52_high, "w52_low": w52_low}
        else:
            logger.debug("52w for %s dropped: %s is not the latest session (%s)", code, target_date, latest)
    return {"w52": w52, "quote": quote}


BASIC_INFO_CACHE_MAX_AGE_DAYS = 30  # 관리종목/거래정지/업종은 자주 안 바뀌므로 매일 재조회하지 않음


def fetch_stock_basic_info(code: str) -> dict | None:
    """Fetch 관리종목/거래정지/업종분류 from KIS 주식기본조회(search_stock_info,
    tr_id CTPF1002R). Reuses the same dsf.search_stock_info call already proven
    in production by probe_nxt_tradeable() above (NXT 판정에 동일 API 사용 중).
    Returns None on any failure - caller keeps the previous cached value (if any).
    """
    row = _search_stock_info_row(code)
    if row is None:
        return None

    admn_item = _is_truthy_flag(row.get("admn_item_yn"))
    tr_stop = _is_truthy_flag(row.get("tr_stop_yn"))
    # [2026-09-23] lstg_stqt(상장주수) - 같은 응답의 기존 미사용 필드. 시가총액(price*lstg_stqt)
    # 정보성 표시용으로만 g002에서 사용 - 상장주식수는 자사주/전략적보유를 포함해 실제 유동주식
    # (float)의 부정확한 근사치이므로 점수에는 반영하지 않는다(g002 changelog 참조).
    lstg_stqty = None
    raw_lstg = row.get("lstg_stqt")
    if raw_lstg not in (None, ""):
        try:
            lstg_stqty = int(float(raw_lstg))
        except (TypeError, ValueError):
            lstg_stqty = None
    return {
        "admn_item_yn": bool(admn_item),
        "tr_stop_yn": bool(tr_stop),
        "sector_large": str(row.get("idx_bztp_lcls_cd_name") or "").strip(),
        "sector_mid": str(row.get("idx_bztp_mcls_cd_name") or "").strip(),
        "sector_small": str(row.get("idx_bztp_scls_cd_name") or "").strip(),
        "lstg_stqty": lstg_stqty,
    }


def _load_basic_info_cache(data_root: Path) -> dict:
    cache_path = data_root / "_stock_basic_info_cache.json"
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("failed to read %s: %s", cache_path, exc)
        return {}


def _save_basic_info_cache(data_root: Path, cache: dict) -> None:
    cache_path = data_root / "_stock_basic_info_cache.json"
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        logger.info("saved stock basic-info cache (%d codes): %s", len(cache), cache_path)
    except Exception as exc:
        logger.warning("failed to write %s: %s", cache_path, exc)


def get_stock_basic_info_cached(cache: dict, code: str, target_date: str) -> dict | None:
    """관리종목/거래정지/업종 - BASIC_INFO_CACHE_MAX_AGE_DAYS 이내에 이미 조회한 종목은
    API를 다시 호출하지 않고 캐시값을 그대로 재사용한다(느리게 바뀌는 참조성 데이터라
    2,557종목 전체를 매일 재조회할 필요가 없음 - RS/52주 고저처럼 날짜별로 값이 바뀌는
    시계열 데이터와는 성격이 다름).
    """
    entry = cache.get(code)
    if entry:
        # [2026-09-23] 스키마 마이그레이션: lstg_stqty(상장주수) 필드가 오늘 새로 추가됐는데,
        # 이 캐시는 코드별 fetched_at 기준 30일 이내면 API를 재호출하지 않고 그대로 반환한다.
        # 실제 캐시 파일을 확인해보니 2,557종목 중 다수가 2026-09-14에 조회돼(9일 전, 30일
        # 이내) "신선"하다고 판정될 상태 - 필드 추가 이전에 저장된 이 항목들은 나이가
        # 30일을 넘기 전까지 lstg_stqty 없이 계속 재사용되어 market_cap이 몇 주간 공란으로
        # 남는다. 신규 필드 키 자체가 없는(구버전 스키마) 항목은 나이와 무관하게 1회 강제
        # 재조회해 채운다 - lstg_stqty 키가 일단 생기면(값이 None이어도) 이후로는 정상적으로
        # 30일 캐시가 적용된다.
        if "lstg_stqty" in entry:
            fetched_at = entry.get("fetched_at")
            if fetched_at:
                # [2026-10-10] 나이는 실제 조회일(KST 오늘) 기준. 이전에는 fetched_at에 대상일을 적고 대상일과
                # 비교해서, 과거 날짜 백필 시 오늘 조회한 값이 과거 날짜로 기록되고 나이도 틀렸음.
                # (이 캐시는 날짜별이 아닌 현재 상태라 과거 날짜 백필에는 여전히 현재 값이 쓰인다.)
                try:
                    age_days = (datetime.strptime(_kst_today(), "%Y%m%d") - datetime.strptime(fetched_at, "%Y%m%d")).days
                except ValueError:
                    age_days = BASIC_INFO_CACHE_MAX_AGE_DAYS + 1
                if 0 <= age_days <= BASIC_INFO_CACHE_MAX_AGE_DAYS:
                    return entry

    fresh = fetch_stock_basic_info(code)
    if fresh is None:
        return entry  # keep stale cache rather than losing the data on a transient API failure
    fresh["fetched_at"] = _kst_today()
    cache[code] = fresh
    return fresh


def fetch_market_index_daily(target_date: str, window: int = 30) -> dict:
    """KOSPI(0001)/KOSDAQ(1001) 최근 `window`거래일 종가 시계열을 KIS
    국내업종 일자별지수(inquire_index_daily_price, tr_id FHPUP02120000)로 조회.
    g002의 RS(상대강도) 계산과 시장 레짐 판정이 지금까지 pykrx(KRX 직접 스크래핑)에
    의존해 네트워크/설치 상태에 따라 조용히 실패하던 것을, 이미 인증된 KIS 세션으로
    대체하기 위함(2026-08-29). 실패 시 빈 dict 반환 - g002는 기존처럼 pykrx로 폴백한다.
    """
    index_fn = getattr(dsf, "inquire_index_daily_price", None)
    if not callable(index_fn):
        return {}

    result: dict[str, list] = {}
    for key, idx_code in (("kospi", "0001"), ("kosdaq", "1001")):
        try:
            _df1, df2 = index_fn(
                fid_period_div_code="D",
                fid_cond_mrkt_div_code="U",
                fid_input_iscd=idx_code,
                fid_input_date_1=target_date,
            )
        except Exception as exc:
            logger.debug("inquire_index_daily_price failed for %s: %s", key, exc)
            continue
        if df2 is None or df2.empty or "bstp_nmix_prpr" not in df2.columns:
            continue

        series_df = df2.copy()
        if "stck_bsop_date" in series_df.columns:
            series_df = series_df.sort_values("stck_bsop_date")
        series_df["_close"] = pd.to_numeric(series_df["bstp_nmix_prpr"], errors="coerce")
        series_df = series_df.dropna(subset=["_close"]).tail(window)
        if len(series_df) >= 2:
            result[key] = series_df["_close"].tolist()
            # [2026-10-10] 날짜를 함께 저장(종가만 있으면 종목 일봉과 날짜 정렬 불가). g002는 kospi/kosdaq
            # 리스트만 읽으므로 별도 키로 추가 - 기존 형식 호환.
            if "stck_bsop_date" in series_df.columns:
                result.setdefault("dates", {})[key] = series_df["stck_bsop_date"].astype(str).tolist()
    return result


def load_symbols(symbols_file: Path) -> list[tuple[str, str]]:
    df = pd.read_csv(symbols_file, comment="#")  # "#" lines = excluded (g010 risk flags / manual)
    if "code" not in df.columns:
        raise ValueError(f"'code' column not found in {symbols_file}")

    names = df["name"].astype(str).tolist() if "name" in df.columns else [""] * len(df)
    pairs: list[tuple[str, str]] = []
    for code, name in zip(df["code"].astype(str), names):
        code6 = code.strip().zfill(6)
        if code6:
            pairs.append((code6, str(name).strip() or code6))

    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for code, name in pairs:
        if code in seen:
            continue
        seen.add(code)
        deduped.append((code, name))
    return deduped


def _load_watchlist_file(path: Path) -> list[tuple[str, str]]:
    """Load code,name pairs from an r004-style watchlist (# comment lines skipped)."""
    pairs: list[tuple[str, str]] = []
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            with open(path, "r", encoding=encoding) as _f:
                for line in _f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(",", 1)
                    code = parts[0].strip().zfill(6)
                    if not code.isdigit() or len(code) != 6:
                        continue
                    name = parts[1].strip() if len(parts) > 1 else code
                    pairs.append((code, name))
            break
        except UnicodeDecodeError:
            continue
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for code, name in pairs:
        if code not in seen:
            seen.add(code)
            deduped.append((code, name))
    return deduped


def _regular_session_last_close(csv_path: Path) -> float | None:
    """Last close of a collected 10s file, restricted to the regular session (<= REGULAR_END).

    KIS's live quote API (`stck_sdpr`/`prdy_clpr`/... used by r006's fetch_prev_close())
    always reports the official regular-session close as "전일 종가", never an NXT
    after-hours print. When a day was collected with --nxt, the file's last row can be
    from the NXT afternoon session (up to 20:00) instead - using it as-is as "prev close"
    would silently diverge from what r006 sees live. Filtering to REGULAR_END keeps both
    in sync regardless of whether --nxt was used for the prior day's collection.
    """
    try:
        _df = pd.read_csv(csv_path, usecols=["datetime", "close"])
    except Exception:
        return None
    _dt = pd.to_datetime(_df["datetime"], errors="coerce")
    _mask = _dt.notna() & (_dt.dt.time <= REGULAR_END) & _df["close"].notna()
    _col = _df.loc[_mask, "close"]
    if len(_col) == 0:
        return None
    return float(_col.iloc[-1])


def _load_daily_closes(output_dir: Path, code: str, name: str) -> pd.Series | None:
    """date(YYYYMMDD str) -> official close (stck_clpr) from {code}_{name}_daily.csv, or None."""
    safe_name = str(name).replace("/", "_").replace("\\", "_")
    path = output_dir / f"{code}_{safe_name}_daily.csv"
    try:
        df = pd.read_csv(path, usecols=["date", "close"], dtype={"date": str})
    except Exception:
        return None
    s = pd.Series(pd.to_numeric(df["close"], errors="coerce").to_numpy(), index=df["date"].str.strip())
    s = s[s > 0]
    return s.sort_index() if not s.empty else None


def _official_prev_close(daily: pd.Series | None, target_date: str) -> tuple[float | None, str | None]:
    """(close, date) of the last daily row before target_date.

    The daily API lists trading days only, so this is the true previous trading day even when
    the prior day's folder was never collected (or collected later) - _compute_prev_close_from_data
    walks back folders and silently returned an older day's close in that case
    (e.g. 20261008 got 20261006's close because 20261007 was collected afterwards).
    The row is final once target_date's collection runs, unlike target_date's own row.
    """
    if daily is None:
        return None, None
    prior = daily[daily.index < target_date]
    if prior.empty:
        return None, None
    return float(prior.iloc[-1]), str(prior.index[-1])


def _official_close_on(daily: pd.Series | None, target_date: str, daily_csv_mtime: float | None) -> float | None:
    """target_date's official close, only if the daily csv was written after that day ended (KST).

    A same-day fetch can still be pre-final (observed: 20261006 fetched 16:38 -> 5420/5.83M shares,
    final 5450/5.95M), so same-day collections keep the minute-bar fallback.
    """
    if daily is None or daily_csv_mtime is None or target_date not in daily.index:
        return None
    day_end = datetime.strptime(target_date, "%Y%m%d").replace(tzinfo=ZoneInfo("Asia/Seoul")) + timedelta(days=1)
    if daily_csv_mtime < day_end.timestamp():
        return None
    return float(daily[target_date])


def _compute_prev_close_from_data(
    code: str, target_date: str, data_root: Path, only_date: str | None = None,
) -> float | None:
    """Return the prior trading day's regular-session close from already-collected data.

    Replicates r006 fetch_prev_close() without extra API calls, so r007 can apply
    MAX_BUY_RISE_PCT_FROM_PREV_CLOSE on stored data. Fallback for symbols without a daily csv:
    when only_date (the known previous trading day) is given, only that folder is accepted.
    """
    import json as _json_pc
    target = datetime.strptime(target_date, "%Y%m%d").date()
    for back in range(1, 15):
        prior = (target - timedelta(days=back)).strftime("%Y%m%d")
        if only_date is not None and prior != only_date:
            continue
        prior_dir = data_root / prior
        if not prior_dir.is_dir():
            continue
        daily_close_path = prior_dir / "_daily_close.json"
        if daily_close_path.exists():
            try:
                with open(daily_close_path, encoding="utf-8") as _f:
                    dc = _json_pc.load(_f)
                val = dc.get(str(code).zfill(6))
                if val is not None:
                    return float(val)
            except Exception:
                pass
        for p in sorted(prior_dir.glob(f"{code}_*_10s.txt")):
            val = _regular_session_last_close(p)
            if val is not None:
                return val
    return None


def _build_prev_close_map(
    symbols: list[tuple[str, str]], target_date: str, output_dir: Path,
    daily_series: dict[str, pd.Series | None] | None = None,
) -> tuple[dict[str, float], int, str | None]:
    """{code: prev close} for target_date -> (map, official count, previous trading day).

    Primary: the daily csv's last row before target_date (official close, true previous trading day).
    Fallback (no daily csv / no prior row): already-collected data, restricted to the previous trading
    day's folder as seen by the other symbols' daily csvs. Unknown previous day -> no fallback at all:
    an older day's close is worse than no value (g003 skips the gate when a code is missing).
    """
    daily_series = daily_series or {}
    prev_close_map: dict[str, float] = {}
    missing: list[str] = []
    prev_days: dict[str, int] = {}
    for code, name in symbols:
        daily = daily_series[code] if code in daily_series else _load_daily_closes(output_dir, code, name)
        val, prev_day = _official_prev_close(daily, target_date)
        if val is None:
            missing.append(code)
            continue
        prev_close_map[code] = val
        prev_days[prev_day] = prev_days.get(prev_day, 0) + 1
    official = len(prev_close_map)

    prev_trading_day = max(prev_days, key=prev_days.get) if prev_days else None
    if missing and prev_trading_day is None:
        logger.warning("prev_close: previous trading day unknown (no daily csv) - %d codes left without prev_close", len(missing))
        return prev_close_map, official, None
    for code in missing:
        val = _compute_prev_close_from_data(code, target_date, output_dir.parent, only_date=prev_trading_day)
        if val is not None:
            prev_close_map[code] = val
    return prev_close_map, official, prev_trading_day


def _parse_code_filter(raw: str) -> set[str]:
    if not raw:
        return set()
    tokens = [part.strip() for part in raw.split(",")]
    return {token.zfill(6) for token in tokens if token}


def resolve_target_date(date_arg: str | None, symbols: list[tuple[str, str]]) -> str:
    if date_arg:
        datetime.strptime(date_arg, "%Y%m%d")
        return date_arg

    # Auto-select latest tradeable date when --date is omitted (weekend/holiday safe).
    probe_symbols = symbols[:8] if symbols else []
    today = datetime.now().date()
    for back in range(0, 14):
        target = (today - timedelta(days=back)).strftime("%Y%m%d")
        for code, _ in probe_symbols:
            try:
                probe_df = fetch_symbol_data(code=code, target_date=target, include_nxt=True)
            except Exception:
                probe_df = None
            if probe_df is not None and not probe_df.empty:
                if back > 0:
                    logger.info("--date omitted. auto-selected latest tradeable date: %s", target)
                return target

    fallback = today.strftime("%Y%m%d")
    logger.warning("could not detect recent tradeable date automatically. fallback to today: %s", fallback)
    return fallback


def main() -> None:
    args = parse_args()
    _install_kis_http_session()

    include_nxt = args.nxt

    symbols_file = Path(args.symbols_file)
    if not symbols_file.is_file():
        raise SystemExit(f"symbols file not found: {symbols_file}")

    if args.watchlist_only:
        symbols = []
    else:
        symbols = load_symbols(symbols_file)

    # Merge extra symbols from --watchlist-file (r004 watchlist)
    if args.watchlist_file:
        wl_paths = [p.strip() for p in args.watchlist_file.split(",") if p.strip()]
        for wl_path_str in wl_paths:
            wl_path = Path(wl_path_str)
            if not wl_path.is_file():
                logger.warning("watchlist-file not found (skipped): %s", wl_path)
                continue
            wl_syms = _load_watchlist_file(wl_path)
            existing_codes = {c for c, _ in symbols}
            added = 0
            for c, n in wl_syms:
                if c not in existing_codes:
                    symbols.append((c, n))
                    existing_codes.add(c)
                    added += 1
            logger.info(
                "watchlist-file %s: %d symbols added (%d total after merge)",
                wl_path.name, added, len(symbols),
            )

    # Parse --date: one or more values, space- and/or comma-separated
    # (e.g. --date 20260508 20260511 / --date 20260508,20260511). Duplicates removed, order kept.
    raw_dates = list(dict.fromkeys(
        d.strip() for arg in (args.date or []) for d in arg.split(",") if d.strip()
    ))
    if raw_dates:
        for d in raw_dates:
            try:
                datetime.strptime(d, "%Y%m%d")
            except ValueError:
                raise SystemExit(f"Invalid --date '{d}'. expected YYYYMMDD")
        target_dates = raw_dates
    else:
        try:
            target_dates = [resolve_target_date(None, symbols)]
        except ValueError:
            raise SystemExit("Could not resolve a target date automatically.")

    ka.auth(svr="prod" if args.env == "real" else "vps")
    selected_codes = _parse_code_filter(args.code)
    if selected_codes:
        symbols = [(code, name) for code, name in symbols if code in selected_codes]
        if not symbols:
            raise SystemExit(f"No matching symbols from --code: {','.join(sorted(selected_codes))}")

    data_root_path = Path(args.data_root)
    basic_info_cache = _load_basic_info_cache(data_root_path)

    for target_date in target_dates:
        output_dir = data_root_path / target_date
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("collect start: date=%s, symbols=%d, include_nxt=%s", target_date, len(symbols), include_nxt)

        # 시장 레짐/RS(상대강도) 계산용 KOSPI/KOSDAQ 일자별 지수 스냅샷 (KIS API, 종목당
        # 아니라 날짜당 1회만 조회). g002가 이 파일을 우선 사용하고, 없으면 기존 pykrx로 폴백.
        market_index = fetch_market_index_daily(target_date)
        if market_index:
            market_index_path = output_dir / "_market_index.json"
            with open(market_index_path, "w", encoding="utf-8") as _f:
                json.dump(market_index, _f, ensure_ascii=False, indent=2)
            logger.info(
                "saved market index snapshot (%s): %s",
                ", ".join(f"{k}={len(v)}d" for k, v in market_index.items() if k != "dates"),
                market_index_path,
            )
        else:
            logger.warning("market index snapshot fetch failed for %s - g002 RS/regime will fall back to pykrx", target_date)

        saved_count = 0
        empty_count = 0
        nxt_flags: dict[str, bool] = {}
        saved_symbols: list[tuple[str, str]] = []
        w52_map: dict[str, dict] = {}
        quote_map: dict[str, dict] = {}

        cutoff_ts = _collection_cutoff_ts(target_date, include_nxt)
        if args.no_resume:
            (output_dir / RESUME_PROGRESS_FILE).unlink(missing_ok=True)
            progress: dict[str, dict] = {}
        else:
            progress = _load_resume_progress(output_dir)
        resumed_count = 0
        adopted_count = 0

        for idx, (code, name) in enumerate(symbols, start=1):

            if not args.no_resume:
                rec = progress.get(code)
                paths = _symbol_output_paths(output_dir, code, name, args.save_legacy_files)
                if not (rec and _resume_record_valid(
                    rec, paths, include_nxt, args.save_legacy_files, cutoff_ts, args.save_10s_indicators,
                )):
                    rec = None
                    if all(p.is_file() for p in paths):
                        rec = _adopt_existing_symbol_files(
                            output_dir, code, name, include_nxt, args.save_legacy_files, cutoff_ts, args.env,
                            target_date, args.save_10s_indicators,
                        )
                        if rec:
                            _append_resume_progress(output_dir, rec)
                            adopted_count += 1
                            logger.info("[%d/%d] %s(%s) | resume: existing files adopted", idx, len(symbols), code, name)
                            if args.sleep > 0:
                                time.sleep(args.sleep)
                if rec and args.flows and rec["status"] == "saved":
                    # Collected without --flows, or some sources came back empty: fetch only the missing sources
                    # (charts are not re-downloaded). Records before per-source tracking: flows=True -> complete.
                    missing = rec.get("flows_missing")
                    if missing is None:
                        missing = [] if rec.get("flows") is True else list(FLOW_SOURCES)
                    attempts = int(rec.get("flows_attempts") or 0)
                    if missing and attempts < FLOW_MAX_ATTEMPTS:
                        missing = fetch_and_save_flows(code, name, target_date, output_dir, tuple(missing))
                        rec = {**rec, "flows": not missing, "flows_missing": missing,
                               "flows_attempts": attempts + 1, "flows_at": time.time()}
                        _append_resume_progress(output_dir, rec)
                        if missing:
                            logger.warning("[%d/%d] %s(%s) | flows missing %s (attempt %d/%d)",
                                           idx, len(symbols), code, name, missing, attempts + 1, FLOW_MAX_ATTEMPTS)
                        if args.sleep > 0:
                            time.sleep(args.sleep)
                if rec:
                    nxt_flags[code] = bool(rec.get("nxt"))
                    if rec["status"] == "saved":
                        if rec.get("w52") and _record_w52_point_in_time(rec, target_date):
                            w52_map[code] = rec["w52"]
                        if rec.get("quote"):
                            quote_map[code] = rec["quote"]
                        get_stock_basic_info_cached(basic_info_cache, code, target_date)
                        saved_count += 1
                        saved_symbols.append((code, name))
                    else:
                        empty_count += 1
                    resumed_count += 1
                    logger.debug("[%d/%d] %s(%s) | resume: skipped", idx, len(symbols), code, name)
                    continue

            nxt_tradeable = False
            df = None
            last_error = None

            # 理쒕? 2???쒕룄
            for attempt in range(2):
                try:
                    nxt_tradeable = probe_nxt_tradeable(code) if include_nxt else False
                    nxt_flags[code] = nxt_tradeable
                    df = fetch_symbol_data(code=code, target_date=target_date, include_nxt=nxt_tradeable)
                    last_error = None
                    break  # ?깃났?섎㈃ 猷⑦봽 ?덉텧
                except Exception as exc:
                    last_error = exc
                    if attempt == 0:
                        logger.warning("[%d/%d] %s(%s) | fetch error (retry): %s", idx, len(symbols), code, name, exc)
                        time.sleep(0.5)  # ?ъ떆?????좉퉸 ?湲?
            # 2???쒕룄 紐⑤몢 ?ㅽ뙣??寃쎌슦
            if last_error is not None:
                logger.error("[%d/%d] %s(%s) | fetch error (final): %s", idx, len(symbols), code, name, last_error)
                nxt_flags[code] = False
                empty_count += 1
                if args.sleep > 0:
                    time.sleep(args.sleep)

                continue

            if df is None or df.empty:

                empty_count += 1
                logger.info("[%d/%d] %s(%s) | NXT=%s | no data", idx, len(symbols), code, name, nxt_tradeable)
                _append_resume_progress(output_dir, {
                    "code": code, "name": name, "status": "empty", "include_nxt": include_nxt,
                    "legacy": args.save_legacy_files, "nxt": nxt_tradeable, "w52": None,
                    "fetched_at": time.time(),
                })
            else:
                df_10s = interpolate_to_10sec(df)

                # [2026-10-10] 지표 열은 기본 생략(--save-10s-indicators로 유지) - g003은 1분/3분봉으로 묶은 뒤
                # r002 calculate_indicators로 다시 계산하고 g009는 종가만 읽어, 파일 용량의 대부분이 미사용이었음.
                if args.save_10s_indicators:
                    df_10s = calculate_r76_indicators(df_10s)
                # amount = 해당 분 거래대금(원), raw_bar==1 행에만 값(보간 행은 공란).
                # raw_bar: 1 = KIS 원본 1분봉 행, 0 = 보간 합성 행. 무체결 분/단일가 구간의 :00 행도 합성이라
                # g003/g009가 :00 행만으로 원본 분봉을 복원하면 가짜 봉이 섞이던 문제 방지용. 두 열 모두 맨 뒤.
                df_10s["amount"] = df_10s.pop("amount")
                df_10s["raw_bar"] = df_10s["datetime"].isin(df["datetime"]).astype("int8")

                safe_name = str(name).replace("/", "_").replace("\\", "_")
                file_10s_path = output_dir / f"{code}_{safe_name}_10s.txt"
                df_10s.to_csv(file_10s_path, index=False, encoding="utf-8-sig", sep=",", float_format="%.2f")

                legacy_log = ""
                if args.save_legacy_files:
                    # [2026-10-10] 3분봉 파일은 읽는 코드가 없어(g003/r003은 직접 3분봉을 만듦) legacy 옵션으로 이동.
                    df_3m = build_3min_indicator_frame(df)
                    file_3m_path = output_dir / f"{code}_{safe_name}_3m.txt"
                    df_3m.to_csv(file_3m_path, index=False, encoding="utf-8-sig", sep=",", float_format="%.2f")
                    legacy_log += f" | 3m={len(df_3m)} -> {file_3m_path}"
                    df_1m = enrich_with_strategy_indicators(df)
                    df_20s = interpolate_to_20sec(df_1m)

                    file_1m_path = output_dir / f"{code}_{safe_name}_1m.txt"
                    legacy_20s_path = output_dir / f"{code}_{safe_name}_20s.txt"

                    df_1m.to_csv(file_1m_path, index=False, encoding="utf-8-sig", sep=",", float_format="%.2f")
                    df_20s.to_csv(legacy_20s_path, index=False, encoding="utf-8-sig", sep=",", float_format="%.2f")
                    legacy_log += (
                        f" | 1m={len(df_1m)} -> {file_1m_path}"
                        f" | 20s(interpolated)={len(df_20s)} -> {legacy_20s_path}"
                    )

                # 일봉 데이터 취득 및 저장 (r002 우하향 종목 필터용). 실패하면 진행기록에 daily=False를 남겨
                # 다음 실행에서 재수집 - 전일/당일 종가와 g002 입력이 이 파일에 의존한다.
                daily_ok = fetch_and_save_daily_ohlcv(
                    code=code, name=name, env_dv=args.env,
                    target_date=target_date, output_dir=output_dir,
                )

                flows_missing = fetch_and_save_flows(code, name, target_date, output_dir) if args.flows else list(FLOW_SOURCES)
                if args.flows and flows_missing:
                    logger.warning("[%d/%d] %s(%s) | flows missing %s - retried on next run", idx, len(symbols), code, name, flows_missing)

                # KIS inquire_price snapshot: real 52-week high/low for r002/g002 high_52w_ratio (dropped when
                # the range was set after target_date) + risk/limit fields for _quote_snapshot.json.
                snap = _quote_record_fields(code, args.env, target_date)
                w52 = snap["w52"]
                if w52:
                    w52_map[code] = w52
                if snap["quote"]:
                    quote_map[code] = snap["quote"]

                # 관리종목/거래정지/업종 스냅샷 (KIS search_stock_info) - 캐시가 신선하면
                # API 재호출 없이 재사용(BASIC_INFO_CACHE_MAX_AGE_DAYS). g002가 하드필터
                # (관리종목/거래정지 배제)와 업종 분산에 사용.
                get_stock_basic_info_cached(basic_info_cache, code, target_date)

                # Written last so a crash mid-symbol leaves no record and the symbol is refetched.
                _append_resume_progress(output_dir, {
                    "code": code, "name": name, "status": "saved", "include_nxt": include_nxt,
                    "legacy": args.save_legacy_files, "nxt": nxt_tradeable, **snap,
                    "daily": bool(daily_ok), "ind10s": bool(args.save_10s_indicators), "flows": args.flows and not flows_missing,
                    **({"flows_missing": flows_missing, "flows_attempts": 1} if args.flows else {}),
                    "fetched_at": time.time(),
                })
                if not daily_ok:
                    logger.warning("[%d/%d] %s(%s) | daily csv fetch failed - will refetch on next run", idx, len(symbols), code, name)

                saved_count += 1
                saved_symbols.append((code, name))
                logger.info(
                    "[%d/%d] %s(%s) | NXT=%s | 10s(interpolated)=%d -> %s%s",
                    idx,
                    len(symbols),
                    code,
                    name,
                    nxt_tradeable,
                    len(df_10s),
                    file_10s_path,
                    legacy_log,
                )

            if args.sleep > 0:
                time.sleep(args.sleep)

        if resumed_count:
            logger.info(
                "resume: %d symbols already collected (%d adopted from existing files) - skipped download",
                resumed_count, adopted_count,
            )

        # Save NXT tradeable flags for live/sim scripts.
        if nxt_flags:
            nxt_flags_path = output_dir / "nxt_flags.json"
            with open(nxt_flags_path, "w", encoding="utf-8") as _f:
                json.dump(nxt_flags, _f, ensure_ascii=False, indent=2)
            logger.info("saved NXT flags (%d codes): %s", len(nxt_flags), nxt_flags_path)

        # Save daily close (regular-session last close) for each symbol, used as next-day
        # prev_close in r007. Restricted to REGULAR_END so --nxt-collected days don't leak
        # an NXT after-hours print in place of the official close (see _regular_session_last_close).
        # Official close (daily csv stck_clpr) is preferred: the minute chart's 15:30 bar can differ from
        # it (20261008: 2,098/2,463 symbols, e.g. 000020 minute 5360 vs official 5410).
        daily_close_map: dict[str, float] = {}
        daily_series: dict[str, pd.Series | None] = {}
        dc_official = 0
        for _dc_code, _dc_name in saved_symbols:
            _safe = str(_dc_name).replace("/", "_").replace("\\", "_")
            _daily = _load_daily_closes(output_dir, _dc_code, _dc_name)
            daily_series[_dc_code] = _daily
            try:
                _daily_mtime = (output_dir / f"{_dc_code}_{_safe}_daily.csv").stat().st_mtime
            except OSError:
                _daily_mtime = None
            _dc_val = _official_close_on(_daily, target_date, _daily_mtime)
            if _dc_val is not None:
                dc_official += 1
            else:
                _f10 = output_dir / f"{_dc_code}_{_safe}_10s.txt"
                _dc_val = _regular_session_last_close(_f10) if _f10.exists() else None
            if _dc_val is not None:
                daily_close_map[_dc_code] = _dc_val
        if daily_close_map:
            daily_close_path = output_dir / "_daily_close.json"
            with open(daily_close_path, "w", encoding="utf-8") as _f:
                json.dump(daily_close_map, _f, ensure_ascii=False, indent=2)
            logger.info(
                "saved daily_close.json (%d codes, %d official / %d minute-bar fallback): %s",
                len(daily_close_map), dc_official, len(daily_close_map) - dc_official, daily_close_path,
            )

        # Written even when empty so a rerun drops a stale file (e.g. an older version's future-leaking 52w).
        quote_path = output_dir / "_quote_snapshot.json"
        if quote_map or quote_path.exists():
            with open(quote_path, "w", encoding="utf-8") as _f:
                json.dump(quote_map, _f, ensure_ascii=False, indent=2)
            pit = sum(1 for q in quote_map.values() if q.get("point_in_time"))
            logger.info("saved quote_snapshot.json (%d codes, %d point-in-time): %s", len(quote_map), pit, quote_path)

        w52_path = output_dir / "_52w_high_low.json"
        if w52_map or w52_path.exists():
            with open(w52_path, "w", encoding="utf-8") as _f:
                json.dump(w52_map, _f, ensure_ascii=False, indent=2)
            logger.info("saved 52w_high_low.json (%d codes): %s", len(w52_map), w52_path)

        # basic_info_cache는 날짜 폴더가 아니라 data_root 최상위에 저장되는 전역 캐시라
        # (관리종목/거래정지/업종은 날짜별로 바뀌는 데이터가 아님) 매 target_date 처리 후
        # 갱신분을 바로 반영 - 여러 날짜를 한 번에 돌리다 중간에 중단돼도 유실을 최소화.
        _save_basic_info_cache(data_root_path, basic_info_cache)

        # Save prev_close from prior trading day data (mirrors r006 fetch_prev_close).
        prev_close_map, pc_official, prev_trading_day = _build_prev_close_map(
            saved_symbols, target_date, output_dir, daily_series,
        )
        prev_close_path = output_dir / "_prev_close.json"
        if not prev_close_map and prev_close_path.exists():
            # A stale file from an earlier (possibly wrong) run must not outlive a rerun that found nothing.
            prev_close_path.unlink()
            logger.warning("prev_close: no values for %s - removed stale %s", target_date, prev_close_path)
        if prev_close_map:
            with open(prev_close_path, "w", encoding="utf-8") as _f:
                json.dump(prev_close_map, _f, ensure_ascii=False, indent=2)
            logger.info(
                "saved prev_close.json (%d codes, %d official / %d folder fallback, prev trading day=%s): %s",
                len(prev_close_map), pc_official, len(prev_close_map) - pc_official, prev_trading_day, prev_close_path,
            )

        # Save date-scoped picks file for r007 simulation input.
        if saved_symbols:
            picks_lines = [f"{code},{name}" for code, name in saved_symbols]
            picks_payload = "\n".join(picks_lines) + "\n"

            underscored_dated_picks = output_dir / f"_{target_date}_picks.txt"

            underscored_dated_picks.write_text(picks_payload, encoding="utf-8-sig")
            logger.info(
                "saved picks file (%d codes): %s",
                len(saved_symbols),
                underscored_dated_picks,
            )

        logger.info("done: date=%s saved=%d, empty=%d, out=%s", target_date, saved_count, empty_count, output_dir)


if __name__ == "__main__":
    main()


