# -*- coding: utf-8 -*-

"""R76 shared configuration for live trading and simulation.

Risk profile presets:
- Set AUTO_TRADING_RISK_PROFILE to one of conservative|neutral|aggressive.
- Korean aliases are supported: 보수|중립|공격.
- If no profile is specified, neutral is applied by default.
- Preset JSON files live under xgraph/auto_trading/risk_profiles/.
"""

import json
import os
from datetime import time as dt_time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path / file constants
# ---------------------------------------------------------------------------
DEFINE_TODAY_CODE_PATH = "r008_trade_watchlist_today.txt"
DATA_DIR_NAME = "data"
# ---------------------------------------------------------------------------
# Feature: R008 watchlist pipeline (scanner -> r008 -> r006 live)
# ---------------------------------------------------------------------------
FEATURE_R008_WATCHLIST = True
FEATURE_SCAN_EXPORT_TO_R008 = True
FEATURE_WATCHLIST_RESOLVE_SCAN_PICKS = True

R008_WATCHLIST_FILENAME = DEFINE_TODAY_CODE_PATH
SCAN_PICKS_LEGACY_FILENAME = "picks.txt"
SCAN_PICKS_PREFIX_TEMPLATE = "_{date}_picks.txt"


# ---------------------------------------------------------------------------
# Indicator parameters
# ---------------------------------------------------------------------------
# 볼린저밴드 기준 기간(봉 개수)
BB_PERIOD = 20
# 볼린저밴드 표준편차 배수(상/하단 폭)
BB_STD_MULTIPLIER = 2.0
# 기본 이동평균 기간(공용)
MA_PERIOD = 5
# 스토캐스틱 %K 계산 기간
STOCH_K_PERIOD = 10
# 스토캐스틱 %D 평활 기간
STOCH_D_PERIOD = 5
# RSI 계산 기간
RSI_PERIOD = 14
# RSI 시그널선 평활 기간
RSI_SIGNAL_PERIOD = 6
# Williams %R 계산 기간
WILLIAMS_R_PERIOD = 10
# Williams %D(평활) 기간
WILLIAMS_D_PERIOD = 9
# MFI 계산 기간
MFI_PERIOD = 14
# 거래량 이동평균 기간
VOLUME_MA_PERIOD = 20
# OBV 이동평균 기간
OBV_MA_PERIOD = 10
# MACD 단기 EMA 기간
MACD_FAST = 5
# MACD 장기 EMA 기간
MACD_SLOW = 12
# MACD 시그널 EMA 기간
MACD_SIGNAL_PERIOD = 4
# ADX 계산 기간
ADX_PERIOD = 7

# ADX 최소값 완화
ADX_MIN_TREND = 15.0  # 20.0 -> 15.0
ADX_STRONG_TREND = 40.0
# 매수 진입용 ADX 최소값
ADX_BUY_MIN = 25.0
# 매수 진입용 MFI 최소값
MFI_BUY_MIN = 50.0
# 매수 시 ADX가 직전봉 대비 우상향이어야 하는지 여부
REQUIRE_ADX_RISING = True
# 매수 시 +DI가 -DI보다 커야 하는지 여부
REQUIRE_DI_PLUS_DOMINANT = True
# MFI 과열 상한(이상일 경우 추격 매수 금지)
MFI_OVERBOUGHT_MAX = 80.0
# RSI 모멘텀 허용 상한(기본: 50~60 구간 유지)
RSI_BUY_MOMENTUM_MAX = 60.0
# OBV 시그널 골든크로스+돌파 확인 필수 여부
REQUIRE_OBV_SIGNAL_CROSS = True

# ATR 계산 기간
ATR_PERIOD = 14
# ATR 기반 손절 배수
ATR_STOP_MULTIPLIER = 1.5
# ATR 기반 익절 배수
ATR_TAKE_PROFIT_MULTIPLIER = 3.0

# ---------------------------------------------------------------------------
# Trading parameters
# ---------------------------------------------------------------------------
# 1회 매수 주문 최대 금액(KRW)
MAX_ORDER_AMOUNT_KRW = 500_000
# 고정 익절 비율(+)
TAKE_PROFIT_PERCENT = 0.025 # +2.5%

# STOP LOSS
STOP_LOSS_PERCENT = -0.021  # 기본 손절 기준(-2.1%)
# 보유 초기 구간에서 사용하는 완화 손절 기준
STOP_LOSS_EARLY_PERCENT = -0.020
# 손절 로직이 본격 적용되기 전 최소 보유 시간(초)
STOP_LOSS_MIN_HOLD_SECONDS = 600

# 추격매수 방지: 전일 종가 대비 현재가 상승률이 임계치 이상이면 매수 차단
MAX_BUY_RISE_PCT_FROM_PREV_CLOSE = 0.23  # 23%

# Post-buy drop early-exit guard (entry-price anchored)
# 매수 직후 급락 방지: 매수 후 일정 시간 동안 현재가가
# 매수가 대비 POST_BUY_BB_DROP_PCT 이상 낮은 상태가
# POST_BUY_DROP_CONFIRM_SECONDS 동안 유지되면 손절 이전에 조기 매도한다.
POST_BUY_BB_DROP_POLLS = 6  # 레거시 로그/호환용 값
POST_BUY_BB_DROP_PCT = 0.007  # 매수가 대비 이탈 임계치 (-0.7%)
POST_BUY_BB_DROP_ARMED_SECONDS = 300.0  # 매수 후 가드 활성 구간 (초, 기본 5분)
POST_BUY_DROP_CONFIRM_SECONDS = 40.0  # 급락 지속 확인 시간 (초)

# 초기 수익 반납 실패 보호
BREAKEVEN_FAIL_ARM_PNL = 0.008  # 한 번이라도 +0.8% 이익 도달 시 활성화
BREAKEVEN_FAIL_GIVEBACK_PCT = 0.0075  # 고점 대비 0.75% 이상 반납
BREAKEVEN_FAIL_CONFIRM_SECONDS = 30.0  # 실패 지속 확인 시간 (초)

# 무추세 시간 손절
NO_TREND_EXIT_ARM_SECONDS = 1200.0  # 20분 동안 추세 미발생 시 점검 시작
NO_TREND_EXIT_MAX_PEAK_PNL = 0.003  # 최고 수익이 +0.3% 미만이면 무추세로 간주
NO_TREND_EXIT_MIN_PNL = -0.005  # 현재 손익이 -0.5% 이하일 때만 적용
NO_TREND_EXIT_CONFIRM_SECONDS = 90.0  # BB 하단 약세 지속 확인 시간 (초)

STARTUP_WARMUP_SECONDS = 90

# 1. 트레일링 임계값 상향
TRAILING_STOP_FROM_PEAK = 0.02  # 2%

# 보조 매도 시그널 score=2일 때 최소 요구 수익률
AUX_SELL_MIN_PNL_SCORE2 = 0.015
# 보조 매도 시그널 score=3일 때 최소 요구 수익률
AUX_SELL_MIN_PNL_SCORE3 = 0.008
# 보조 매도 시그널 score=4일 때 최소 요구 수익률
AUX_SELL_MIN_PNL_SCORE4 = 0.003

# MA5-BB 하향 크로스 시 즉시 매도 허용 손익 기준
MA5_BB_DOWN_CROSS_IMMEDIATE_PNL = -0.007
# MA5-BB 하향 크로스 즉시 매도 최소 점수
MA5_BB_DOWN_CROSS_IMMEDIATE_SCORE = 2
# MA5-BB 하향 크로스 확정 최소 점수
MA5_BB_DOWN_CROSS_CONFIRM_MIN_SCORE = 1
# MA5-BB 하향 크로스 매도 허용 최소 손익
MA5_BB_DOWN_CROSS_MIN_PNL = 0.000

# 박스권(횡보) 구간에서는 기술적 매도 홀드 기능 사용
ENABLE_BOX_RANGE_HOLD_TECH_SELL = True
# 익절 후 연장 트레일링 기능 사용 여부
ENABLE_TP_EXTENSION_TRAILING = True
TP_EXTENSION_TRAIL_FROM_PEAK = 0.004  # 0.4%
# 박스권 판정 시 참조할 최근 봉 개수
BOX_RANGE_HOLD_LOOKBACK_BARS = 8
# 박스권 판정용 가격 범위 상한
BOX_RANGE_HOLD_MAX_RANGE_PCT = 0.0065
# 박스권 판정용 BB 폭 상한
BOX_RANGE_HOLD_MAX_BB_WIDTH_PCT = 0.0080

# Near-cross ARM 모드: BB 중단과 MA5 간 최대 허용 갭
NEAR_CROSS_ARM_GAP_MAX = 0.0045
# Near-cross ARM 모드: MA5 최소 상승률
NEAR_CROSS_ARM_MA_RISE_MIN = 0.0006
# Early near-cross 모드: BB 중단과 MA5 간 최대 허용 갭
NEAR_CROSS_EARLY_GAP_MAX = 0.0045
# Early near-cross 모드: MA5 최소 상승률
NEAR_CROSS_EARLY_MA_RISE_MIN = 0.0010
# ARM 상태가 유효한 최대 봉 수
NEAR_CROSS_ARM_EXPIRE_BARS = 2

# Near-cross ARM 진입 로직 사용 여부
ENABLE_NEAR_CROSS_ARM = True
# 장초반 early near-cross 진입 로직 사용 여부
ENABLE_EARLY_NEAR_CROSS_ENTRY = True
# 가격 선행 돌파(price-lead breakout) 진입 로직 사용 여부
ENABLE_PRICE_LEAD_BB_BREAKOUT = True
# price-lead breakout 최소 보조지표 점수
PRICE_LEAD_BREAKOUT_MIN_SCORE = 3
# price-lead breakout 최소 ADX
PRICE_LEAD_BREAKOUT_MIN_ADX = 25.0
# 과열 상태에서도 price-lead breakout 허용 여부
PRICE_LEAD_BREAKOUT_ALLOW_OVERBOUGHT = True

# early near-cross 진입 허용 시작 시각
EARLY_NEAR_CROSS_ALLOWED_START = dt_time(9, 0)
# early near-cross 진입 허용 종료 시각
EARLY_NEAR_CROSS_ALLOWED_END = dt_time(11, 30)
# NXT 세션에서 early near-cross 허용 여부
EARLY_NEAR_CROSS_ALLOW_NXT = False
# early near-cross 절대 거래량 최소치
EARLY_NEAR_CROSS_MIN_VOLUME = 800
# early near-cross 거래량 MA 최소치
EARLY_NEAR_CROSS_MIN_VOL_MA = 500
# early near-cross 최소 거래대금(KRW)
EARLY_NEAR_CROSS_MIN_TURNOVER_KRW = 5_000_000

# 폴링 주기
# 메인 루프 폴링 간격(초)
POLL_INTERVAL_SECONDS = 10  # 15 -> 10 (더 빠른 대응)
# 실시간 현재가 재조회 간격(초)
LIVE_PRICE_POLL_INTERVAL_SECONDS = 10
# BB 크로스 판정용 버퍼(노이즈 필터)
LIVE_PRICE_BB_BUFFER_PCT = 0.0005  # 0.0008
# 상향 크로스 확정에 필요한 연속 관측 횟수
LIVE_PRICE_CROSS_CONFIRM_POLLS = 3
LIVE_PRICE_CROSS_CONFIRM_SECONDS = 10  # 20 -> 10 (보다 빠른 진입 확인)
# 하향 크로스 확정에 필요한 연속 관측 횟수
LIVE_PRICE_DOWN_CROSS_CONFIRM_POLLS = 1
# 하향 크로스 확정에 필요한 최소 유지 시간(초)
LIVE_PRICE_DOWN_CROSS_CONFIRM_SECONDS = 0
# 계좌/체결 상태 동기화 주기(초)
ACCOUNT_SYNC_INTERVAL_SECONDS = 90

# 진입 전에 필요한 최소 확정 봉 개수
MIN_BARS_REQUIRED = 3
# 당일 같은 종목 재매수 허용 여부
ALLOW_REBUY_SAME_CODE = False
# 동일 종목 재진입 쿨다운(분)
TRADE_COOLDOWN_MINUTES = 3

# ---------------------------------------------------------------------------
# Session / time constants
# ---------------------------------------------------------------------------
# NXT 세션 활성화 여부 및 시간 설정
ENABLE_NXT_SESSION = False  # NXT 세션 포함 운용 여부
MORNING_NXT_START = dt_time(8, 0)
MORNING_NXT_END = dt_time(8, 50)
REGULAR_START = dt_time(9, 0)
REGULAR_END = dt_time(15, 30)
# 신규 진입 허용 종료 시각(정규장)
REGULAR_NEW_ENTRY_CUTOFF = dt_time(15, 20)
# 강제 청산 시작 시각(정규장)
REGULAR_FORCE_EXIT = dt_time(15, 20)
AFTERNOON_NXT_START = dt_time(15, 30)
AFTERNOON_NXT_END = dt_time(20, 0)
# 신규 진입 허용 종료 시각(오후 NXT)
AFTERNOON_NXT_NEW_ENTRY_CUTOFF = dt_time(19, 59)
# 강제 청산 시작 시각(오후 NXT)
AFTERNOON_NXT_FORCE_EXIT = dt_time(19, 59)

# ---------------------------------------------------------------------------
# Filter coefficients
# ---------------------------------------------------------------------------
STOCH_OVERBOUGHT = 96.0  # 85.0 -> 92.0 -> 96.0 (과열 차단 기준 완화)
# 매수 스토캐스틱 K 하한
STOCH_BUY_MIN = 20.0
# 매수 스토캐스틱 K 상한
STOCH_BUY_MAX = 50.0
# 매수 RSI 하한
RSI_BUY_MIN = 50.0
# 매수 RSI 상한
RSI_BUY_MAX = 70.0
# 매수 허용 Williams %R 하한
WILLIAMS_BUY_FLOOR = -70.0

# 2. Williams R 완화
WILLIAMS_OVERBOUGHT_CEIL = -10  # -20 -> -10
BB_UPPER_PROXIMITY_MAX = 1.05  # 0.85 -> 1.05 (BB 상단에서 충분히 진입 허용)
# BB 폭 최소치(너무 좁은 횡보 구간 배제)
BB_SQUEEZE_MIN_WIDTH_PCT = 0.0
# OBV 돌파 판정에 사용할 과거 봉 개수
OBV_BREAKOUT_LOOKBACK_BARS = 5

# 효과: 라이브 가격 크로스만으로도 진입 가능
ENABLE_STRICT_MA5_BB_GOLDEN_CROSS = False  # True -> False
# 강추세일 때 과열 필터 일부 우회 허용
ENABLE_STRONG_TREND_OVERBOUGHT_BYPASS = True
# 과열 우회 허용 최소 보조지표 점수
STRONG_TREND_OVERBOUGHT_MIN_SCORE = 2
# 과열 우회 허용 최소 거래량 비율
STRONG_TREND_OVERBOUGHT_MIN_VOL_RATIO = 1.00
# 과열 우회 허용 최소 ADX
STRONG_TREND_OVERBOUGHT_MIN_ADX = 15.0
# 추격매수 방지: 실시간 BB 상향 크로스 없이(신호 없음) MA5/BB 후행 진입 시
# 현재가가 BB 중심선 대비 과도하게 이격되면 매수 차단
MA5_BB_FOLLOW_CHASE_MAX_GAP_PCT = 0.002  # 0.20% -- tightened: buy only when price is within 0.2% of BB middle

# BB 중앙선 상승 돌파 전략 파라미터 (BB slope break cross strategy)
BB_SLOPE_LOOKBACK_BARS = 20      # BB 기울기 측정 봉 수 (3분봉 기준 약 1시간)
BB_UPPER_GAP_MIN_PCT = 0.25      # BB 상단 여유 최소치 (%) - 상단까지 여유 없으면 매수 차단 (0.5->0.25)
CANDLE_GAIN_MIN_PCT = 0.1        # 현재봉 양봉 최소 상승률 (%) - 약세봉 진입 차단 (0.3->0.1)
BB_BUY_SCORE_THRESHOLD = 8       # 매수 최소 점수 (공격형=6, 중립형=8, 보수형=10)

# 3. 거래량 완화
# 장초반 거래량 필터 비율
VOLUME_RATIO_OPEN = 0.40  # 0.60 -> 0.40 (아침 변동성 높은 시간 더 완화)
# 한낮 거래량 필터 비율
VOLUME_RATIO_MIDDAY = 0.35  # 0.45 -> 0.35 (한낮 변동성 낮은 시간 더 완화)
# 장마감 구간 거래량 필터 비율
VOLUME_RATIO_CLOSE = 0.50  # 0.70 -> 0.50 (마감 거래량 필터 완화)
# NXT 세션 거래량 필터 비율
VOLUME_RATIO_NXT = 0.30  # 0.40 -> 0.30 (NXT 세션 거래량 필터 더 완화)
# 강추세일 때 거래량 필터 완화 가중치
VOLUME_RATIO_STRONG_RELAX = 0.15  # 0.10 -> 0.15 (강한 추세 보너스 증가)
# 거래량 필터 최소 하한선
VOLUME_RATIO_FLOOR = 0.50

# 시장일 확인 실패 시 보수적으로 비거래 처리
MARKET_DAY_FAIL_CLOSED = True
# 세션 종료 컷오프에서 전량 청산 강제 여부
SESSION_FORCE_CLOSE_ALL_AT_CUTOFF = True
# 세션 종료 홀드 예외 허용 여부
ENABLE_SESSION_EXIT_HOLD_WITHIN_STOP = False
# 계좌-감시종목 불일치 로그 출력 최소 간격(초)
WATCHLIST_MISMATCH_LOG_INTERVAL_SECONDS = 300

# 오전 NXT 신규 진입 종료 시각(별도 변수)
MORNING_NXT_NEW_ENTRY_CUTOFF = MORNING_NXT_END


# ---------------------------------------------------------------------------
# Live execution parameters (r006_trade_live_execute)
# ---------------------------------------------------------------------------
# 3분봉 프레임 갱신 주기(초)
FRAME_POLL_INTERVAL_SECONDS = 20
# 시작 시 과거 바 백필 동기화 범위(초)
FRAME_BACKFILL_SYNC_SECONDS = 600
# 매수 연속 확인 횟수
BUY_CONSECUTIVE_CONFIRM_COUNT = 2
# 주문 상태 폴링 간격(초)
ORDER_STATUS_POLL_INTERVAL_SECONDS = 15
# 현재가 조회 backoff 초기값(초)
LIVE_PRICE_BACKOFF_BASE_SECONDS = 5
# 현재가 조회 backoff 최대값(초)
LIVE_PRICE_BACKOFF_MAX_SECONDS = 60
# 현재가 stale TTL(초)
LIVE_PRICE_STALE_TTL_SECONDS = 20
# 미결 주문 상태 backoff 최대값(초)
PENDING_STATUS_BACKOFF_MAX_SECONDS = 120
# 메인 루프 연속 오류 허용 최대 횟수
MAIN_LOOP_MAX_CONSECUTIVE_ERRORS = 20
# 라이브 상태 저장 주기(초)
LIVE_STATE_SAVE_INTERVAL_SECONDS = 60
# 미결 매수 주문 대기 여유 시간(초)
PENDING_BUY_GRACE_SECONDS = 90

# 인트라바 실시간 진입 필터
ENABLE_INTRABAR_LIVE_ENTRY_FILTER = True
INTRABAR_MIN_ELAPSED_SECONDS = 90.0
INTRABAR_MFI_MIN = 50.0
INTRABAR_MFI_MAX = 75.0
INTRABAR_RSI_MIN = 50.0
INTRABAR_RSI_MAX = 70.0
INTRABAR_ADX_MIN = 20.0

# AUX 매도 슬리피지 버퍼
AUX_SELL_MIN_REALIZED_TARGET_PCT = 0.010
AUX_SELL_TRIGGER_SLIPPAGE_BUFFER_PCT = 0.005

# ---------------------------------------------------------------------------
# Simulation parameters (r007_trade_simulate_by_date)
# ---------------------------------------------------------------------------
# 초기 시뮬레이션 자본금(KRW)
SIM_INITIAL_CAPITAL = 5_000_000
# 매도 완료 후 재진입 허용 여부
SIM_ALLOW_REENTRY_AFTER_COMPLETED_SELL = True
# 웜업용 tail 봉 개수
SIM_WARMUP_TAIL_BARS = 160
# 웜업 이전 데이터 최대 조회일수
SIM_WARMUP_PRIOR_MAX_DAYS = 20
# 기술적 매도 최소 보유 시간(초)
TECH_SELL_MIN_HOLD_SECONDS = 300
# 당일 최소 처리 봉 수
SAME_DAY_MIN_BARS = 10
# 시뮬레이션 내부 체크 간격(초)
SIM_CHECK_INTERVAL_SECONDS = 10
# 10초 그리드 시뮬레이션 기본값
SIMULATE_10S_GRID_DEFAULT = True
# 인트라바 볼륨 폴백 활성화
ENABLE_INTRABAR_VOLUME_FALLBACK = True
# 인트라바 볼륨 폴백 최소 진행률
INTRABAR_VOLUME_FALLBACK_MIN_PROGRESS = 0.30

# 시뮬레이션 완화 게이트
SIM_RELAXED_SHARED_GATES = True
SIM_RELAXED_MIN_SUPPORT_SCORE = 0
SIM_RELAXED_VWAP_MAX_UNDER_PCT = 0.0025
SIM_RELAXED_REQUIRE_CONFIRMED_ABOVE = False
SIM_RELAXED_REQUIRE_MA5_BIAS = False
SIM_RELAXED_ALLOW_BELOW_BB = True
SIM_RELAXED_ALLOW_FALLING_TREND = True


def _apply_risk_profile_overrides() -> None:
	raw_profile = (os.environ.get("AUTO_TRADING_RISK_PROFILE") or os.environ.get("RISK_PROFILE") or "").strip()
	if not raw_profile:
		raw_profile = "neutral"

	alias = {
		"safe": "conservative",
		"conservative": "conservative",
		"보수": "conservative",
		"balanced": "neutral",
		"neutral": "neutral",
		"중립": "neutral",
		"aggressive": "aggressive",
		"attack": "aggressive",
		"공격": "aggressive",
	}
	profile = alias.get(raw_profile.lower())
	if profile is None:
		print(
			"[WARN] Unknown AUTO_TRADING_RISK_PROFILE "
			f"'{raw_profile}'. Expected conservative|neutral|aggressive (or 보수|중립|공격)."
		)
		return

	profile_path = Path(__file__).resolve().parent / "risk_profiles" / f"{profile}.json"
	if not profile_path.is_file():
		print(f"[WARN] Risk profile file not found: {profile_path}")
		return

	try:
		overrides = json.loads(profile_path.read_text(encoding="utf-8"))
	except Exception as exc:
		print(f"[WARN] Failed to load risk profile '{profile}': {exc}")
		return

	if not isinstance(overrides, dict):
		print(f"[WARN] Invalid risk profile format (expected object): {profile_path}")
		return

	applied_keys: list[str] = []
	for key, value in overrides.items():
		if key in globals():
			globals()[key] = value
			applied_keys.append(key)

	print(f"[INFO] Applied risk profile '{profile}' ({len(applied_keys)} overrides)")


_apply_risk_profile_overrides()

