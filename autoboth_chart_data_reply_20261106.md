# 회신: AutoBotH 주가 차트·보조지표용 서버 데이터 (요청 2026-11-06 → 처리 2026-11-06)

> 서버 저장소(`auto_trading_src`) 쪽 회신이다. 요청 문서 "서버 데이터 요청 프롬프트: AutoBotH 주가 차트·보조지표용 수집 데이터"의
> P1~P3 항목 순서로 답하고, 마지막에 요청된 회신 4가지(확인 답변 / 새 파일·열·예시 / 새 명령 출력 예시 / 보관·용량·CPU)를 모은다.
> 모든 새 명령은 `r016_control.py`의 하위 명령이며 읽기 전용, stdout JSON 하나, rc 0(성공)/2(입력·파일 없음)다. 호출은 기존과 같이
> `cd /data/srv/autobot/auto_trading_src && ~/venv/bin/python live_trading/r016_control.py <command> ...`.

---

## P1-1. 틱 파일의 체결 구분 → `venue`, `session` 열 추가 (완료)

**확인 답변**
1. 틱 파일은 **KRX 체결만** 들어 있다(g011은 KIS 웹소켓 `H0STCNT0` = KRX 실시간체결; NXT는 `H0NXCNT0`, 통합은 `H0UNCNT0`로 구독하지 않는다).
   08:30~08:59 체결(예: 003490 08:30:05~08:30:55, 6건, 전부 전일 종가 29,600, `ccld_dvsn=3`, 호가 0)은 **KRX 장전 시간외 종가매매**(08:30~08:40, 전일 종가 단일가)다.
   NXT 프리마켓이 아니다.
2. 두 값의 정체가 다르다.
   - 15:30:11 틱 30,350 (`cntg_vol` 103,834, `ccld_dvsn` 공란) = **15:20~15:30 종가 동시호가 체결** = KRX 정규장 종가. 같은 날짜 폴더의
     `_daily_close.json["003490"]`도 30,350이고, g001의 `_10s.txt` 15:30 원본 분봉 종가도 30,350이다.
   - `20261105/003490_대한항공_daily.csv`의 20261105 행 종가 30,100 = g001이 **같은 날 16:56에 KIS 일봉 API(`inquire_daily_itemchartprice`, 시장 J)로
     받은 당일 행**이다. g001 Update log(2026-10-10)에 적혀 있듯 **당일 폴더의 당일 행은 "미확정 관측"**이고(2026-10-06 사례: 16:38 5,420 → 최종 5,450),
     다음 거래일 폴더의 같은 날짜 행이 공식 값이다. 실제로 20261104 행은 11/04 폴더 거래량 2,688,494 → 11/05 폴더 2,839,048로 바뀌었고,
     20261103 행 종가는 11/03 폴더 29,050 → 11/04 폴더 29,150으로 바뀌었다(KIS 일봉 API가 시간외·NXT 거래를 뒤늦게 반영하는 것으로 보인다. 원인은 KIS 쪽이라 확정 못 함).
   - **봇이 쓰는 값**: r003 실전은 KIS 현재가 조회의 기준가 `stck_sdpr`(전일 공식 종가)를 전일 종가로 쓴다(r005 `_018`, r012 예열 기준가). g002 스캐너와 g003
     시뮬은 `_prev_close.json`(= 그 날짜 폴더 `_daily.csv`의 **직전 행**, 즉 공식 종가)을 쓴다. 15:30 틱/분봉 종가(30,350)는 봇이 "전일 종가"로 쓰지 않는다.
   - **앱 권장**: 과거 일봉은 다음날 이후 폴더(또는 아래 P3-2 이력 파일에서 `source_date > date`인 행)를 쓰고, **당일 봉은 틱(정규장+동시호가)으로 만들어
     종가 = 동시호가 체결가**로 표시한다. 이력 파일의 `source_date == date` 행은 "미확정"으로 표시한다.

**구현** (`data_simulation/g011_tick_collector.py`, `live_trading/r009_tick_bars.py`, 커밋 0e539d0)
- `TICK_FILE_COLUMNS` v2 = 기존 13열 + `venue`, `session` (끝에 추가, 기존 열 순서 불변).
  - `venue`: 항상 `KRX`.
  - `session`: 체결시각 vs r014 달력의 정규장 (open, close). `pre` = open 전, `regular` = [open, close), `close_auction` = [close, close+10분)
    (15:20~15:30 동시호가는 체결이 없고 15:30:xx에 한 번 찍힌다), `post` = 그 뒤(장후 시간외). 수능일(10:00~16:30)도 달력대로 따라간다.
- **기존 파일 호환**: 수집기가 장중 재시작해 이미 13열 헤더로 시작된 파일에 이어 쓸 때는 그 파일의 헤더 폭(13열)으로 계속 쓴다. r009/r013 판독기가
  "헤더와 폭이 다른 행"을 끊긴 행으로 버리기 때문에 한 파일에 13/15열이 섞이면 안 된다. 따라서 **앱은 헤더 이름으로 열을 찾고, `session`이 없으면
  시각으로 구분**(지금 하는 임시 처리)하는 폴백을 유지해 달라. 15열은 수집기가 재시작된 다음 날짜 폴더의 새 파일부터 생긴다(2026-11-06 08:50 cron 기동분부터).
- 비종목 파일: `ticks/YYYYMMDD/_collector.log`(수집기 로그; `START`, `CODES a,b,...`, 재접속 기록) 하나뿐이고, `ticks/.g011.lock`은 날짜 폴더 밖의 단일 실행 락이다.
  `^\d{6}\.csv$`만 종목으로 보는 규칙은 맞다.

## P1-2. 10초 파일 형식 (답변 (a) + 1분봉은 `bars --kind min`으로 제공)

**확인 답변** (`g001_data_collect_symbols_daily.py` docstring 9~12행, `interpolate_to_10sec`, r012 `causal_minute_frames`)
- `raw_bar=1` 행의 시각은 **봉 시작**이다. `2026-11-05 09:00:00` 행 = 09:00:00~09:00:59 체결로 만든 분봉이고(틱으로 검증: 그 분의 첫 체결 31,000 / 고 31,700 /
  저 30,850 / 마지막 31,500 = 10초 파일의 O/H/L/C와 일치), 봇은 이 봉을 09:01:00에 볼 수 있다(r012는 종료 라벨 = 시작+1분으로 바꿔 쓴다).
  마지막 `2026-11-05 15:30:00` 행은 종가 동시호가 체결(15:30:11) 하나로 만든 봉이다.
- `volume`은 **그 분 거래량 ÷ 6**(10초 행 하나의 몫)이다. `raw_bar=1` 행도 ÷6 값이다(09:00 행 62,275 × 6 = 373,650 ≈ 틱 합계 372,764; 틱 수집 누락분 차이).
  소수 2자리로 반올림되며 1,000 이상이면 정수로 반올림된다. 분 거래량 = 그 분 6행의 합 또는 `raw_bar` 행 × 6(근사). `amount`는 `raw_bar=1` 행에만 있고
  **그 분 전체 거래대금(원)**이다. 보간 행(`raw_bar=0`)의 OHLC는 직선 보간 값이라 호가 단위에 맞지 않는다(57,767 같은 값).
- 08:50~09:00, 15:20~15:30 구간의 :00 행은 보간으로 생긴 가짜 행이다(r012 `session_gap_mask`가 제외).
- `market` 열 `J` = KRX 분봉. g001은 NX(NXT) → J → UN 우선순위로 분을 채우고 같은 분이 겹치면 NX가 이긴다(`include_nxt` 옵션 시). 현재 수집은 J만 보인다.

**(b) 별도 1분봉 파일은 만들지 않았다.** 대신 아래 `bars --kind min`이 틱 파일이 있으면 틱으로 정확한 M분봉을, 없으면 `_10s`의 `raw_bar=1` 행으로 1분봉(거래량 ×6)을 돌려준다.
파일을 하나 더 유지하는 것보다 요청 시 계산 + 캐시가 Pi 디스크·I/O에 낫다고 판단했다. 꼭 파일이 필요하면 말해 달라(r017 함수를 그대로 써서 10분이면 만든다).

## P1-3. 틱 보관 기간과 대상

**확인 답변**
- **삭제하는 코드가 없다.** g011/g022/g024 어디에도 보관 기간 상수나 정리 로직이 없다. 5일치만 있는 이유는 **틱 수집을 2026-10-30에 시작**했기 때문이다
  (`runtime/data/ticks/20261030`이 첫 날). 앞으로는 계속 쌓이며 20거래일 이상 자연히 보관된다. 하루 20~70 MB(종목 수에 비례)라 월 ~1 GB 수준이고 디스크(468 GB 중 320 GB 여유)에 문제 없다.
  압축(gzip)은 r013이 전일 파일 꼬리를 읽는 경로(`iter_prior_tail`)까지 바꿔야 해서 지금은 하지 않는다. 필요해지면 "N일 지난 폴더 gzip" 작업을 따로 만들겠다.
- 수집 대상: `g022_tick_collect_daily.sh`(cron 08:50/08:55, 장중 5분마다 생존 확인)가 `g011 --fill-from-scan`으로 띄운다 = **r004 감시목록 종목 전부 + 가장 최근 g002 스캔 순위 상위 후보**를
  합쳐 **최대 40종목**(KIS 웹소켓 세션당 구독 상한). 그래서 하루 20~40종목이다. r003도 g011이 없으면 자기 감시목록으로 띄운다(`TICK_COLLECTOR_AUTOSTART`).
  감시목록 밖 종목의 틱은 없고, 분봉은 `_10s`(전 종목 ~3,000개)로 대신하는 지금 방식이 맞다.

## P2-1. 조건별 프레임 → `conditions` 출력에 `frame` 추가 (완료, 커밋 7e3b4c0)

**확인 답변**
1. 봉 크기: 매수 트리거(`_008`)는 **1분봉 MA5/MA20**(`TRIPLE_1MIN_CONDITION="above"`) + **60/120틱봉 골든크로스**(`TRIPLE_TICK_CONDITION="any"`, r013)의 3중 조건(`ENTRY_TRIGGER_MODE=triple`).
   컨텍스트 조건(`_009/_014/_015/_017`)과 매도의 ATR/반전/같은 봉 판정(`_002/_003/_015/_019/_021/_022`)은 **3분봉**(1분봉 3개 집계, 종료 라벨 = r003 `buy_frame`/`frame`).
   `_029`는 지수 ETF(069500/229200)의 1분봉 MA5>MA20. `_018/_028`은 전일 종가(일봉, KIS `stck_sdpr`). PRE_ORDER 나머지와 매도 손절/익절/트레일/이익잠금은 **현재가 폴링(live)**.
2. 예열: **포함한다.** `ENABLE_PRIOR_DAY_INDICATOR_WARMUP=True`, `PRIOR_DAY_WARMUP_MINUTES=180` → 전 거래일 원본 1분봉 마지막 180개(그 3분 집계 = 60개 전후)를 당일 봉 앞에 붙여
   지표(BB/RSI/MACD/ADX/EMA)를 계산하고 당일 행만 쓴다(r002 `calculate_indicators_with_warmup`; VWAP/OBV는 당일만). 단 전일 마지막 종가와 오늘 기준가가 0.3% 넘게 다르면(권리락 등) 예열을 쓰지 않는다.
   앱의 "전일 120봉 예열"은 1분 기준이면 180으로 맞추면 봇과 같다. 3분봉은 전일 3분봉 60개.

**출력**: `conditions`의 각 조건에 `"frame"`: `none | live | 1m | 3m | tick | daily`, 복수면 `+`로 연결(`"1m+tick"`, `"live+3m"`, `"3m+daily"`). 최상위에 `"frames"`(설명)와
`"warmup": {"prior_day_1m_bars": 180, "enabled": true, "note": ...}`. r015 단독 실행도 같다.

## P2-2. 봇이 계산한 지표 값 → `indicators` (완료, 커밋 84b45ca)

```
r016_control.py indicators --date YYYYMMDD --code CODE --frame 1m|3m|daily [--last N]
```
- 봇과 **같은 입력 파일**(그 날짜 폴더 `_10s.txt`의 `raw_bar=1` 원본 1분봉, 전일 폴더의 예열 180분봉, 기준가 = 그 날짜 폴더 `_prev_close.json`, 없으면 `_daily.csv` 직전 행)과
  **같은 함수**(r012 `causal_minute_frames`/`load_prior_day_warmup`, r002 `calculate_indicators_with_warmup`)로 다시 계산한다. r003 메모리는 건드리지 않는다.
  당일 3분봉은 r003 `_normalize_intraday_frame`과 같이 첫 부분봉만 버리고 15:30 동시호가 봉(15:33 라벨)은 유지한다. 1분 프레임은 r003이 당일 봉을 최근 200개만 들고
  계산하므로(`FRAME_CACHE_MAX_BARS`) 200봉을 넘는 행은 같은 200봉 창으로 다시 계산해 돌려준다 — 단 창 재계산은 Pi에서 봉당 0.16 s라 **요청한 행(`--last N`)이 40개를 넘으면
  한 번에 계산**하고 `"parity": "single_pass ..."`로 알린다(EMA/RSI/ADX가 소수점 수준으로 다를 수 있음). `--last 40` 이하면 `"parity": "exact"`.
  실전 r003은 KIS 분봉 API를 직접 받고 기준가를 KIS `stck_sdpr`에서 받으므로 입력이 100% 동일하지는 않다(권리락·분할일엔 r003이 예열을 끄는데 여기서는 켜질 수 있음 -
  `warmup` 문자열에 기준가 출처를 적는다). g003 시뮬과는 동일 경로다.
- **장중(그날 `_10s` 파일이 아직 없을 때)**은 틱 파일의 정규장 체결로 1분봉을 만들어 같은 지표를 계산한다(`source`에 `ticks -> 1m, live approximation`). 틱 누락분만큼 KIS 분봉과
  거래량이 조금 다를 수 있다. 장 마감 후 g001이 돌면 자동으로 `_10s` 경로로 바뀐다.
- 행 = `[ts(봉 시작), ts_end(봉 종료 = 봇이 그 봉을 보는 시각), open, high, low, close, volume, MA_5, EMA_20, EMA_60, VOL_MA20, BB_MIDDLE, BB_UPPER, BB_LOWER, BB_STD,
  RSI, RSI_SIGNAL, STOCH_K, STOCH_D, WILLIAMS_R, WILLIAMS_D, MACD, MACD_SIGNAL, MACD_HIST, DI_PLUS, DI_MINUS, ADX, ATR, MFI, VWAP, OBV, OBV_MA]`
  (`columns`에 실제 순서가 온다). 숫자는 소수 6자리 이하, 결측 `null`. `params`에 r001 기간 상수, `warmup`에 예열 사유 문자열.
- `--frame daily`는 `_daily.csv`의 지표 열을 그대로 준다(일봉 지표는 g001이 HTS 기본 기간 MACD 12/26/9, ADX 14 등으로 계산 - r001 장중 기간과 다름).

실제 출력(003490, 2026-11-05, 3분봉 마지막 행; 앞 7열 + 일부만 표시):
```json
{"ok": true, "frame": "3m", "code": "003490", "date": "20261105",
 "source": "/data/srv/autobot/runtime/data/20261105/003490_대한항공_10s.txt",
 "warmup": "prior_day=20261104 1min=180 3min=59 (base_price from _prev_close.json; r003 live uses KIS stck_sdpr)", "parity": "exact",
 "params": {"MA_PERIOD": 5, "BB_PERIOD": 20, "BB_STD_MULTIPLIER": 2.0, "RSI_PERIOD": 14, "RSI_SIGNAL_PERIOD": 6, "MACD_FAST": 5, "MACD_SLOW": 12, "MACD_SIGNAL_PERIOD": 4, "ADX_PERIOD": 7, "PRIOR_DAY_WARMUP_MINUTES": 180, ...},
 "columns": ["ts", "ts_end", "open", "high", "low", "close", "volume", "MA_5", ..., "BB_MIDDLE", "BB_UPPER", "BB_LOWER", "BB_STD", "RSI", "RSI_SIGNAL", ..., "MACD", "MACD_SIGNAL", "MACD_HIST", "DI_PLUS", "DI_MINUS", "ADX", "ATR", "MFI", "VWAP", "OBV", "OBV_MA"],
 "rows": [["2026-11-05T15:15:00", "2026-11-05T15:18:00", 30400, 30400, 30350, 30350, 18415.8, 30330, ..., 30350, 30435.839508, 30264.160492, 42.919754, 50.769771, 48.069348, ..., 2.955732, -4.705178, 7.66091, 17.0939, 4.045509, 38.707388, 91.686498, 44.057457, 30360.913476, 177802.2, 189618.72]],
 "elapsed_ms": 1030}
```
오류 예: `indicators --date 20250101 --code 003490 --frame 3m` → rc 2, `{"ok": false, "error": "no _10s file (raw_bar rows) for 003490 on 20250101"}`

## P3-1. 서버 쪽 봉 집계 → `bars` (완료, 커밋 84b45ca)

```
r016_control.py bars --date YYYYMMDD --code CODE --kind tick|min|day|week|month [--n N] [--ticks T=60] [--interval M=1] [--warmup K] [--since-ts ISO] [--all-sessions]
```
- `tick`: g011 틱 CSV에서 T틱 봉(마지막 봉은 미완성일 수 있음). `min`: 틱 CSV가 있으면 틱으로 M분봉, 없으면 `_10s` 원본 분봉(거래량 ×6, `source`에 파일명으로 구분).
  기본은 `session`이 `regular`/`close_auction`인 체결만(열이 없는 v1 파일은 시각으로 같은 구분); `--all-sessions`면 장전/장후 포함.
  틱 중복(재접속 재전송)은 `acml_vol` 기준으로 제거하고, 헤더 폭과 다른 행(끊긴 행)은 r009/r013과 같이 버린다. r009 `TickBarBuilder`의 갭/역순 규칙과는 미세하게 다를 수 있다
  (차트용 집계이지 r013 트리거 재현이 아님).
- `day|week|month`: P3-2의 `daily_history/<code>.csv`가 있으면 그것(2025-09~), 없으면 날짜 폴더 `_daily.csv`(200행). 주봉 `t` = 그 주 월요일, 월봉 `t` = 1일.
- `--since-ts`: 그 시각 이후 봉만(마지막 봉 포함). 응답의 `next_since`를 다음 호출에 넘기면 된다. `--warmup K`: 직전 거래일(최대 10일 거슬러 찾음)의 마지막 K봉을 `warm`에, 그 날짜를 `warm_date`에.
  단 전일 틱 파일이 없으면(그날 감시 종목이 아니었으면) tick 종류는 `warm`이 비고, min 종류는 `_10s`로 채운다.
- **캐시**: `runtime/cache/bars/YYYYMMDD/<code>_<tag>.json`에 집계 결과 + **파일 바이트 오프셋 + 미완성 마지막 봉의 체결**을 저장한다. 다음 호출은 파일이 자란 만큼만 파싱해 이어 붙이고,
  파일이 그대로면 캐시만 돌려준다(`source`에 `(cache)`). 지난 날짜도 같은 구조. 캐시는 파일의 (dev, inode)와 오프셋 직전 256바이트 지문을 함께 저장해, 파일이 교체되거나
  줄었다가 다시 자라면 처음부터 다시 만든다.

실제 출력(003490, 2026-11-05, 60틱봉 마지막 2개):
```json
{"ok": true, "kind": "tick", "code": "003490", "date": "20261105", "source": "/data/srv/autobot/runtime/data/ticks/20261105/003490.csv",
 "ticks": 60, "interval": null, "sessions": "regular+close_auction", "columns": ["t", "o", "h", "l", "c", "v"],
 "bars": [["2026-11-05T15:19:29", 30400, 30400, 30300, 30350, 5992], ["2026-11-05T15:19:38", 30350, 30350, 30250, 30350, 104708]],
 "count": 2, "warm": [], "warm_date": null, "next_since": "2026-11-05T15:19:38", "elapsed_ms": 2275}
```
(`t` = 봉의 첫 체결 시각. 마지막 봉에 15:30:11 동시호가 103,834주가 들어가 있다.)
1분봉 증분 호출 예: `bars --date 20261105 --code 003490 --kind min --since-ts 2026-11-05T15:18:00` →
`"bars": [["2026-11-05T15:18:00", 30350, 30400, 30350, 30350, 9955], ["2026-11-05T15:19:00", 30400, 30400, 30250, 30300, 14335], ["2026-11-05T15:30:00", 30350, 30350, 30350, 30350, 103834]]`
오류 예: `bars --date 20261105 --code 999999 --kind tick` → rc 2, `{"ok": false, "error": "no tick file / _10s file for 999999 on 20261105"}`

## P3-2. 일봉 이력 파일 (완료, 커밋 d78f0d0, 서버에서 1회 재구성 완료)

- `runtime/data/daily_history/<code>.csv`, 열 `date,open,high,low,close,volume,amount,source_date`. `source_date` = 그 행을 제공한 날짜 폴더.
  **`source_date == date`인 행은 미확정(그날 16시대 관측)**, `source_date > date`면 다음날 이후 폴더의 공식 값이다. 같은 날짜가 여러 폴더에 있으면 나중 폴더가 이긴다.
- `data_simulation/g025_daily_history.py --rebuild`(종목별로 그 종목 파일이 있는 폴더 중 처음/100폴더 간격/마지막 2개를 읽어 재구성, Pi에서 약 5분) /
  `--append [YYYYMMDD]`(그 날짜 폴더를 upsert; 생략 시 최신 폴더).
  장 마감 후 g001이 끝난 뒤 돌려야 한다. g001은 cron이 아니라 수동 실행이라, 서버 crontab에 `0 21 * * 1-5 ~/venv/bin/python .../g025_daily_history.py --append >> runtime/log/g025_cron.log 2>&1`
  추가를 사용자에게 요청해 두었다(앱 쪽과 무관).
- 범위: 날짜 폴더가 2026-07-01부터이고 각 파일이 200행이라 **2025-09-03부터**다(003490: 285행, 약 14개월). **3년은 지금 데이터로 안 된다.** g001의 `lookback_days`를
  늘려 한 번 더 받으면(KIS 일봉 API, 전 종목 3년 ≈ 종목당 8회 호출, 장 마감 후 20~30분) 채울 수 있다 - 필요하면 그 작업을 따로 하겠다. 주봉은 지금 약 62개, 월봉 15개.
- 3,045종목, 총 49 MB(날짜 폴더 하나 1.1~1.4 GB와 비교). 기존 날짜 폴더 파일은 그대로 둔다.

예시 3행(003490):
```
date,open,high,low,close,volume,amount,source_date
20250903,23500,23750,23350,23450,621917,14602245800,20260701
20261104,29050,30100,28650,29600,2839048,83139399975,20261105
20261105,31000,31700,29550,30100,3889393,118239474200,20261105   ← 미확정(source_date == date)
```

## P3-3. 틱 수집기 상태 → `status.collector` (완료, 커밋 9d34cf0/84b45ca)

`status`(와 `collector status`)에 아래 블록이 있다. `stale`은 장중(09:00~15:30, 평일)에만 채워지고 감시목록 종목 중 60초 넘게 기록이 없는 종목이다.
`missing`은 감시목록에 있는데 돌고 있는 g011이 구독하지 않는 종목(감시목록을 바꾼 뒤 생김 → `collector restart`).
```json
"collector": {"running": true, "pid": 1234, "pids": [1234], "started_at": "2026-11-06T08:50:01", "subscribed": ["003490", "..."], "symbols": 23,
              "last_write": "2026-11-06T09:12:31", "stale": [{"code": "161390", "last_write": null}], "missing": [],
              "ticks_dir": "/data/srv/autobot/runtime/data/ticks/20261106"}
```
(요청 필드명 `symbols`/`last_write`/`stale` 그대로. 지금(야간)은 `running:false`, `symbols:0`, `last_write:null`.)

---

## 회신 요청 4가지 요약

1. **확인 답변**: P1-1 틱은 KRX만, 08:30대는 장전 시간외 종가매매, 당일 `_daily.csv` 행은 미확정 관측이고 공식 종가는 다음날 폴더 행(봇의 전일 종가 = KIS `stck_sdpr` / `_prev_close.json`),
   15:30:11 틱 = 동시호가 종가. P1-2 `raw_bar=1` 시각 = 봉 시작, `volume` = 분 거래량/6(주), `amount` = 분 거래대금(원). P1-3 삭제 로직 없음(10/30부터 수집 시작), 대상 = 감시목록 + 스캔 후보 최대 40종목.
   P2-1 트리거 1분봉+60/120틱, 컨텍스트·ATR·반전 3분봉, 예열 전일 1분봉 180개(3분봉 60개).
2. **새 파일**: 틱 CSV 열 `venue,session` 추가(새 파일부터, 기존 파일은 13열 유지); `runtime/data/daily_history/<code>.csv`(위 예시 3행); `runtime/cache/bars/`(서버 내부 캐시, 앱이 읽지 않음).
3. **새 명령 출력**: 위 P2-2(`indicators` 정상/오류), P3-1(`bars` 정상/증분/오류). `conditions`에 `frame`/`frames`/`warmup` 추가.
4. **보관·용량·CPU**: 틱은 무기한 보관(월 ~1 GB), 일봉 이력 49 MB. Pi(라즈베리파이 5) 측정: 틱 5.3 MB(57,943체결) 첫 집계 **2.9 s**(`bars --kind tick`, csv 폭 검사 포함), 이후 증분 호출
   **평균 0.07 s**(9회 분할 호출 합계 2.9 s), 파일 변화 없을 때 캐시 응답 **0.03 s**; `indicators --frame 3m` **0.9 s**, `--frame 1m --last 1` **1.3 s**, `--last 30` **5.5 s**(200봉 창 재계산),
   `--frame 1m` 전체(381행) **1.1 s**(single_pass); `bars --kind day` 0.1 s 미만. 장중 10초 폴링은 `--since-ts` 증분 호출만 쓰면 봇 루프에 영향 없는 수준이다.
   `indicators`는 폴링하지 말고 노드/종목을 눌렀을 때 1회 호출(`--last 40` 이하 권장)을 권한다.

변경 파일: `data_simulation/g011_tick_collector.py`, `live_trading/r009_tick_bars.py`, `live_trading/r015_export_conditions.py`, `live_trading/r016_control.py`,
`live_trading/r017_chart_data.py`(신규), `data_simulation/g025_daily_history.py`(신규), 테스트 5개(전체 348개 통과). Codex 검토 1회(6건 반영: 캐시 교체 감지, 기준가 출처, 200봉 창, 15:30 3분봉, 끊긴 행, 종목별 재구성).
