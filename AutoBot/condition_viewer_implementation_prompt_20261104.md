# 구현 가이드 프롬프트: r003 매수/매도 조건 파이프라인 뷰어 (Windows, Electron + npm)

> 이 문서는 그대로 코딩 에이전트(또는 개발자)에게 주는 프롬프트다. "너는 ~를 구현한다" 형식으로 썼고, 필요한 로그 문법·조건 목록·매핑 표를 모두 포함한다.
> 서버 쪽 준비물(조건 JSON 내보내기 스크립트 `live_trading/r015_export_conditions.py`)은 이미 저장소에 있다.

---

## 0. 당신의 임무

리눅스 서버(라즈베리파이)에서 돌아가는 KRX 자동매매 봇 **r003**의 로그를 **SSH로 읽어** Windows 데스크톱 앱에서
**조건 파이프라인 다이어그램**으로 보여주는 프로그램을 만든다.

- 봇은 감시 종목마다 10~30초 주기로 매수 조건을 **번호 순서대로** 평가한다. 하나라도 실패하면 거기서 멈추고 `[REJECT]`(또는 `[BUY SKIP]`,
  `[BUY HOLD]`) 로그를 남긴다. 모두 통과하면 `[BUY EVAL] OK` → `[BUY SUBMIT]` → `[BUY PENDING]`/`[BUY EXEC]` → `>>> Buy` 블록이 찍힌다.
- 보유 종목은 매도 조건을 순서대로 평가해 `[SELL HOLD]`(보류) 또는 `[SELL TRIGGER]` → `[SELL SUBMIT]` → `[SELL FILLED]` → `>>> Sell` 블록을 남긴다.
- 앱은 **읽기 전용**이다. 주문을 내지 않고 서버 파일을 쓰지 않는다. 수십 초 지연은 허용된다.

사용자가 보고 싶은 것: "A 종목이 조건 1~8까지 통과하다가 _014 ATR 하한에서 막혔다", "오늘 전 종목이 어느 조건에서 가장 많이 막혔나",
"보유 종목 B는 어느 매도 조건이 발동했나/보류 중인가".

## 1. 기술 스택 (고정)

| 항목 | 선택 | 비고 |
|---|---|---|
| 런타임 | Electron 33+, Node 20+ | Windows x64 설치 파일(NSIS) |
| UI | React 18 + TypeScript + Vite | `electron-vite` 템플릿 권장 |
| 다이어그램 | `@xyflow/react` (React Flow) + `dagre` 자동 배치 | 노드=조건, 엣지=평가 순서 |
| 차트 | `recharts` | 깔때기/타임라인 |
| 상태 | `zustand` | 렌더러 전역 상태 |
| SSH/SFTP | `ssh2` (메인 프로세스에서만) | 렌더러는 IPC로만 접근 |
| 설정 저장 | `electron-store` | 호스트/포트/사용자/키 경로/원격 루트/폴링 주기 |
| 테스트 | `vitest` | 파서 단위 테스트 필수 |
| 빌드 | `electron-builder` | `npm run build:win` |

보안 규칙: 비밀번호/개인키는 OS 사용자 홈의 설정 파일에만 저장(평문 금지, `safeStorage` 사용). 렌더러에 `nodeIntegration` 금지, `contextIsolation` 유지, preload로 IPC만 노출.

## 2. 서버의 데이터 (전부 읽기 전용)

원격 루트(설정값, 기본): `/data/srv/autobot` — 서버 코드는 `real_trading_bot/AutoBot/`, 런타임은 `runtime/`.

### 2.1 로그 디렉터리 `runtime/log/YYYYMMDD/`

| 파일 | 내용 | 주의 |
|---|---|---|
| `r003_trade_live_execute_YYYYMMDD_HHMMSS.log` | 메인 로그(모든 종목, 모든 태그). r003이 **재시작할 때마다 새 파일** | 하루에 여러 개. mtime 최신 파일을 tail하고 나머지는 1회 전체 파싱 |
| `r003_trade_live_execute_YYYYMMDD_HHMMSS_buy_sell.log` | 그 실행분의 매매 요약(아래 2.3) | |
| `YYYYMMDD_buy_sell.log` | 하루 전체 매매 요약(재시작과 무관하게 누적) | 체결 결과의 1차 소스 |
| `YYYYMMDD_<code>_<name>.txt` | 종목별 로그(메인 로그 중 그 종목 코드가 들어간 줄만) | 종목 타임라인은 이 파일이 가장 쉽다 |
| `YYYYMMDD_change_history.txt` | 감시목록 변경 이력 | 선택 |
| `g011_tick_collector.log` | 틱 수집기 | 선택 |

- 인코딩 UTF-8, 시각은 KST(서버 로컬). 파일은 append-only. 한 줄이 쓰이는 도중에 읽힐 수 있으므로 **마지막 불완전 줄은 다음 폴링으로 넘긴다**.
- 하루 메인 로그는 5~9만 줄(5~10 MB). 반드시 **바이트 오프셋 증분 읽기**(SFTP `stat` 크기 → `createReadStream({start: offset})`)로 처리한다.
- 일자 변경/재시작 감지: 폴링마다 디렉터리 목록을 읽어 새 메인 로그가 생기면 그 파일을 새 오프셋 0으로 추가한다.

### 2.2 메인 로그 한 줄의 문법

```
2026-11-04 09:08:12,593 [INFO]   [BUY EVAL    ] 159010_아스플로   | OK TRIPLE_CROSS_1MIN_MA5_ABOVE_MA20+TICK120_GOLDEN_CROSS+HYBRID_3MIN_CTX_SCORE_12 | 09:07:56 | LIVE 29,050 | ...
```

정규식(한 줄):
```
^(?<date>\d{4}-\d{2}-\d{2}) (?<time>\d{2}:\d{2}:\d{2}),(?<ms>\d{3}) \[(?<level>\w+)\]\s+\[(?<tag>[A-Z][A-Z0-9 _-]*?)\s*\] (?<body>.*)$
```
- `tag`는 공백 패딩이 있는 고정폭 라벨(`BUY EVAL    `) → 트림해서 쓴다.
- `body`의 첫 토큰이 `^(?<code>\d{6})_(?<name>\S*)\s*\|` 꼴이면 종목 줄이다(이름은 로그용으로 잘릴 수 있음: `카카오게~`). 종목명은 `WATCH | 159010 | 아스플로 | NXT=False` 줄이나 `YYYYMMDD_<code>_<name>.txt` 파일명에서 얻는다.
- 태그가 없는 줄(`R003 START ...`, `WATCH | ...`, `REENTRY state restored ...`, `HARD_STOP codes restored ...`)도 있다 → 태그 없는 줄은 별도 처리.
- 메인 로그 외에 숫자 천 단위 쉼표(`29,050`)가 들어가므로 숫자 파싱 때 쉼표 제거.

### 2.3 매매 요약 로그 `YYYYMMDD_buy_sell.log`

```
2026-11-04 09:08:12 [INFO] [BUY SUBMIT  ] 159010_아스플로   | qty=8 | price=29,050 | market | KRX | order_no=0003529700 | reason=TRIPLE_CROSS_...
2026-11-04 09:09:26 [INFO] ==============================================================================================================
2026-11-04 09:09:26 [INFO] >>> 159010_아스플로 Buy - reason=TRIPLE_CROSS_1MIN_MA5_ABOVE_MA20+TICK120_GOLDEN_CROSS+HYBRID_3MIN_CTX_SCORE_12 signal=None live=29,050 bb_mid=28485.0 ...
2026-11-04 09:09:26 [INFO] *** 매수시각=2026-11-04 09:07:56 | 매수가격=29,124 | 매수수량=17
2026-11-04 09:15:56 [INFO] >>> 159010_아스플로 Sell - 수익금액=-10,413원 | 수익율=-2.10% | reason=HARD_STOP_LOSS_1.7PCT exch=KRX
2026-11-04 09:15:56 [INFO] *** 매수시각=2026-11-04 09:07:56 | 매수가격=29,124 | 매수수량=17
2026-11-04 09:15:56 [INFO] *** 매도시각=2026-11-04 09:15:56 | 매도가격=28,511 | 매도수량=17
2026-11-04 10:50:37 [INFO] [REENTRY LIVE] 017900_광전자    (광전자) | BUY qty=55 price=8,970 used=1/1 | reason=...
```
- 시각 형식이 메인 로그와 다르다(밀리초 없음). `>>> {code}_{name} (Buy|Sell) - ` 줄 + 다음 `*** ` 줄들을 하나의 거래 이벤트로 묶는다.
- 부분 매도(`TP1_PARTIAL_40PCT_...`)는 같은 매수에 Sell 블록이 여러 번 나온다. 수익금액은 원 단위 정수(쉼표 포함, 음수 가능).

### 2.4 조건 목록(진실의 원천) - 서버에서 JSON으로 받는다

SSH exec: `cd /data/srv/autobot/real_trading_bot/AutoBot && ~/venv/bin/python live_trading/r015_export_conditions.py`
→ stdout에 JSON 하나:
```json
{"generated_at": "...", "entry_trigger_mode": "triple", "trading_strategy_mode": "normal",
 "buy":  [{"order":0,"no":1,"func":"_001_entry_time_window","stage":"STATE","title":"신규 진입 허용 시간대","description":"...","defines":[...],"values":{...}}, ...],
 "sell": [{"order":0,"no":1,"func":"_001_stale_live_price_guard","stage":"GUARD", ...}, ...]}
```
- `order`가 **평가 순서**다(번호 `no`와 다르다 - 예: _025/_026이 _008 앞, _029가 _009 앞). 다이어그램은 `order`로 배치한다.
- 앱 시작 시 1회 + "새로고침" 버튼으로 다시 받는다. 실패하면 마지막 성공본을 캐시에서 쓴다(앱에 내장 스냅샷 1부 동봉).
- 2026-11-04 현재 매수 25개 / 매도 15개. 현재 매수 평가 순서:
  `_001 _002 _003 _004 _005 _006 _025 _026 | _008 | _029 _009 _014 _015 _017 | _018 _019 _020 _031 _030 _028 _027 _021 _022 _023 _024`
  (STATE | TRIGGER | CONTEXT | PRE_ORDER). 매도: `_001 _002 _003 _004 _006 _007 _023 _015 _016 _017 _018 _019 _020 _021 _022`.

## 3. 로그 → 조건 매핑 (파서 핵심)

### 3.1 매수 경로 - "어느 조건에서 멈췄나"

판정 규칙: 종목의 한 폴링에서 아래 태그 중 **마지막 줄**이 현재 상태다. 각 줄은 하나의 조건에 매핑된다.

| 태그 / body 접두어 | 매핑 조건 | 상태 |
|---|---|---|
| `[SKIP ]` body `can_trade_code_now=False` | `_001` | 시간대 밖(개장 전/후) |
| `[SKIP ]` body `frame=None` / `bars=N < INDICATOR_WARMUP_BARS` | (조건 전 단계) 프레임 준비 | 데이터 대기 |
| `[REJECT]` `SESSION_OPEN_WARMUP` | `_002` | |
| (로그 없음) | `_003` 쿨다운, `_004` 같은 봉 중복 | 조용히 건너뜀 - 표시만 "silent" |
| `[BUY SKIP]` `ALREADY_TRADED_TODAY_UNTIL_SELL` | `_005` | 보유/주문 중 |
| `[BUY SKIP]` `HARD_STOP_REENTRY_BLOCKED_TODAY` | `_006` | 당일 손절 종목 |
| `[BUY SKIP]` `DAILY_LOSS_LIMIT ...` | `_025` | |
| `[BUY SKIP]` `MAX_OPEN_POSITIONS ...` | `_026` | |
| `[REJECT]` `TRIPLE_CROSS_MISSING_...` | `_008` 하위단계 **trigger** | 1분 MA/틱 크로스 없음·만료·틱 파일 없음·저밀도(`TICK_LOW_DENSITY_`) |
| `[REJECT]` `TRIPLE_CROSS_1MIN_CHASE_BUY_BB_GAP_...` / `..._CHASE_BUY_INTRABAR` / `..._CANDLE_NOT_BULLISH_...` | `_008` 하위단계 **candle/gap guard** | |
| `[REJECT]` `TRIPLE_CROSS_1MIN_LIVE_DROP_...` / `..._1MIN_LIVE_BELOW_BB_MID` | `_008` 하위단계 **live guard** | |
| `[REJECT]` `TRIPLE_CROSS_CHASE_FROM_CROSS_x%_GT_y%` | `_008` 하위단계 **chase-from-cross** | 2026-11-04 신규 |
| `[REJECT]` `HYBRID_1MIN_TRIGGER_...` / `HYBRID_1MIN_FRAME_UNAVAILABLE` | `_008` (legacy 경로) | ENTRY_TRIGGER_MODE=legacy 때만 |
| `[TICK TRIG]` `no tick file before 09:00` | `_008` | 개장 전 1회 안내(반려 아님) |
| `[REJECT]` `MARKET_REGIME_DOWN_...` | `_029` | 2026-11-04 신규 |
| `[REJECT]` `HYBRID_3MIN_CTX_INSUFFICIENT_BARS` / `HYBRID_3MIN_CTX_MISSING_INDICATOR` | `_009` | |
| `[REJECT]` `HYBRID_3MIN_CTX_LOW_ENTRY_ATR...` | `_014` | |
| `[REJECT]` `HYBRID_3MIN_CTX_LOW_VOL_MA_ABS_` / `_LOW_ABS_VOLUME_` / `_LOW_VOLUME_RATIO_` / `_LOW_TURNOVER_` | `_015` | |
| (반려 없음, 사유에 `HYBRID_3MIN_CTX_SCORE_n` 기록) | `_017` | 점수는 정보 |
| `[REJECT]` `EXCESSIVE_RISE_FROM_PREV_CLOSE_...` | `_018` | |
| `[REJECT]` `ANTI_CHASE_...` (`ANTI_CHASE_VWAP_GAP_3.95%_GE_3.00%` 등) | `_019` | |
| `[REJECT]` `OPENING_GAP...` | `_020` | |
| `[REJECT]` `QUOTE_STATUS_BLOCK_...` | `_031` | 2026-11-04 신규 |
| `[REJECT]` `ACCOUNT_SYNC_STALE ...` | `_030` | 2026-11-04 신규 |
| `[EDGE AI]` `p=0.372 thr=0.310 (pass|would_block|block) mode=...` 그리고 `[REJECT]` `EDGE_AI_LOW_PROB_p=..._LT_...` | `_028` | `[EDGE AI]` 줄은 정보(확률), `[REJECT]`가 실제 차단 |
| `[REJECT]` `SPREAD_NO_QUOTE` / `SPREAD_TOO_WIDE ...`, `[SPREAD]` `quote query failed ... pass` | `_027` | |
| `[BUY HOLD]` `reason=WAIT_NEXT_POLL_CONFIRM | count=1/2` | `_021` | 연속 확인 대기(통과 중) |
| `[REJECT]` `INSUFFICIENT_BUYING_POWER_OR_BUDGET` | `_022` | |
| `[REJECT]` `STALE_LIVE_PRICE` | `_023` | |
| `[REJECT]` `ORDERBOOK_ASK_THIN` | `_024` | |
| `[BUY EVAL]` `OK <buy_reason> | HH:MM:SS | LIVE ...` | 전부 통과 | 주문 직전 |
| `[BUY SUBMIT]` / `[BUY PENDING]` / `[BUY PARTIAL]` / `[BUY PART CXL]` / `[BUY RP CNCL|RSUB|QUIT|ABAND]` / `[BUY EXEC]` / `[BUY NOFILL]` | 주문 상태 | 다이어그램 끝 "주문" 노드 |
| `[PENDING]` `side=BUY|SELL qty=... submitted=HH:MM:SS` | 주문 대기 중 | 평가 건너뜀 |
| `[REJECT]` body에 `| STEPS 1min_trigger(P|F|-) min_entry_atr_volatility(P|F|-) min_liquidity_safety(P|F|-)` | 진단 | 반려된 폴링에서도 다른 게이트의 P/F를 보여줌 - 노드 색에 쓴다 |

파싱 팁: `[REJECT]` body = `{사유} | GATES ... | STEPS ...`. 사유는 첫 `|` 앞, 접두어 매칭은 위 표 순서대로 **가장 긴 접두어 우선**. 알 수 없는 사유는 "unknown" 노드로 보내고 원문을 보존한다(새 조건이 추가돼도 앱이 깨지면 안 된다).

### 3.2 매도 경로

| 태그 / 사유 | 매핑 조건 |
|---|---|
| `[SELL HOLD]` `AUX_BLOCKED_SCORE_n_PNL_x%_LT_y%` / `AUX_TRIGGER_BUFFER_BLOCK ...` | `_022` 반전 매도가 수익 요건 미달로 보류 |
| `[SELL TRIGGER]` `HARD_STOP_LOSS_1.7PCT` | `_004` |
| `TP1_PARTIAL_40PCT_x.xxPCT[_SURGE]` | `_006` |
| `TP2_SURGE_LADDER_...` / `TP3_SURGE_LADDER_...` | `_007` |
| `PROFIT_LOCK_100PCT_GIVEBACK_0.4PCT` / `PROFIT_LOCK_REST_BREAKEVEN` | `_023` |
| `ATR_TAKE_PROFIT_3.0x` (`[TP_EXTENSION]` 줄은 정보) | `_015` |
| `POST_BUY_ENTRY_DROP_...` | `_016` |
| `BREAKEVEN_FAIL_...` | `_017` |
| `NO_TREND_EXIT_...` | `_018` |
| `ATR_STOP_LOSS_1.5x` | `_019` |
| `TRAILING_STOP_GIVEBACK_...` / `TP_EXTENSION_TRAIL_1.0%` | `_020` |
| `AUX_REVERSAL_SCORE_n` / `LIVE_PRICE_BB_DOWN_CROSS_CONFIRMED_n` | `_022` |
| `PEAK_NEXT_BAR_BEARISH_...` | `_003` |
| `REGULAR_CLOSE_SESSION_FORCE_CLOSE_ALL_...` / `NXT_CLOSE...` | 세션 종료 강제 청산(조건 밖) |
| `[SELL SUBMIT]` → `[SELL FILLED] filled=17/17 | price=28,511 | remaining=0 | pnl=-2.10% | reason=...` → `[SELL EXEC]` | 주문 상태 |
| `[SEL RSUB ERR]` / `submission_unknown` | 접수 불명 복구 중 |

### 3.3 감시목록/시장 상태 (보조)

`[WL SPLIT] active=17 backup=0`, `[ACT PROMOTE]`, `[ACT DROPOUT] reason=TIME_LIMIT_HARD`, `[ACT GRADUATE] reason=POSITION_OPENED`, `[ACT REENTER] used=0/1`,
`[MKT REGIME] KOSPI -> UP|DOWN|N/A | ...`(2026-11-04 신규), `[TICK CROSS] 028260 | tick120 MA5>MA20 at 09:00:57 ...`, `[CHECK]`(폴링마다, 지표 스냅샷 - 무시해도 됨).

## 4. 데이터 모델 (렌더러)

```ts
type Stage = 'STATE'|'TRIGGER'|'CONTEXT'|'PRE_ORDER'|'ORDER';
interface Condition { order:number; no:number; func:string; stage:Stage; title:string; description:string; values:Record<string,unknown>; }
interface EvalEvent { ts:number; code:string; tag:string; condFunc:string|'ORDER'|'UNKNOWN'; sub?:string; result:'pass'|'fail'|'hold'|'info'; reason:string; raw:string; steps?:Record<string,'P'|'F'|'-'> }
interface SymbolState { code:string; name:string; market?:string; watch:'active'|'backup'|'graduated'|'dropped'; position?:{qty:number; buyPrice:number; buyTs:number; peakPct?:number};
  lastEval?:EvalEvent; furthestOrderToday:number; blockCounts:Record<string,number>; timeline:EvalEvent[]; trades:Trade[] }
interface Trade { code:string; side:'BUY'|'SELL'; ts:number; price:number; qty:number; reason:string; pnlKrw?:number; pnlPct?:number }
```
- `furthestOrderToday`: 오늘 그 종목이 도달한 최대 `order`(통과한 마지막 조건). `[BUY EVAL] OK`면 전체 통과.
- `blockCounts[func]`: 오늘 그 조건에서 막힌 폴링 수. 전 종목 합계가 깔때기 그래프다.
- 메인 프로세스가 파싱하고 렌더러에는 **집계된 스냅샷**(종목별 상태 + 최근 N개 이벤트)만 IPC로 보낸다. 원문 줄은 디스크(앱 데이터 폴더, 일자별 SQLite 또는 JSONL)에 두고 요청 시 조회.

## 5. 화면 (4개 뷰 + 설정)

1. **파이프라인 다이어그램** (메인): 매수 조건 25개 노드를 `order` 순서로 세로 배치, 단계(STATE/TRIGGER/CONTEXT/PRE_ORDER/ORDER)별 색 띠.
   - 종목 선택 시: 통과한 노드 초록, 막힌 노드 빨강(사유 툴팁 + 원문), 도달 못 한 노드 회색, `_008`은 하위단계(trigger/guard/live/chase) 서브노드로 펼침.
   - 종목 미선택 시: 노드마다 오늘 "막힌 횟수 / 통과 횟수" 배지 + 노드 크기를 막힌 횟수에 비례(깔때기). `_017`처럼 반려 없는 노드는 점선.
   - 매도 파이프라인은 탭 전환(보유 종목 선택 시 `[SELL HOLD]`/`[SELL TRIGGER]` 매핑).
2. **종목 보드**: 감시 종목 표 - 현재 상태(마지막 평가 시각, 막힌 조건, 보유 여부, 손익), 정렬/필터(막힌 조건별), 클릭하면 1번 뷰 연동.
3. **타임라인**: 선택 종목의 하루 - x축 시각, y축 조건 `order`(도달 깊이), 점 색=결과, 매수/매도 마커. 호버에 원문 로그.
4. **매매 결과**: `YYYYMMDD_buy_sell.log` 기반 거래 표 + 일 누적 손익 라인 + 매도 사유별 합계(막대).
5. **설정/연결**: SSH(호스트·포트·사용자·키/비밀번호), 원격 루트, 폴링 주기(기본 10초), 날짜 선택(과거 날짜는 폴링 없이 1회 로드), 연결 상태 표시, "조건 JSON 새로고침".

상단 상태바: 연결 상태, 마지막 수신 시각, 오늘 파일 수/오프셋, 파싱 경고 수(unknown 사유 등), `[MKT REGIME]` 현재값.

## 6. 동작 요구사항

- 시작: 설정 로드 → SSH 연결 → 조건 JSON 수신 → 오늘 `runtime/log/YYYYMMDD/` 목록 → 메인 로그 전체 1회 파싱(워커 스레드) → 이후 10초마다 증분.
- 재연결: 네트워크 끊김 시 지수 백오프(5s→60s), 오프셋은 유지. 파일 크기가 줄면(회전) 0부터 다시.
- 날짜 경계: 서버 KST 기준 자정에 새 디렉터리로 전환. 앱은 "오늘" = 서버 날짜(`date +%Y%m%d` exec로 확인).
- 성능: 10만 줄 파싱 2초 이내(워커), 렌더러 메모리 300MB 이하, 타임라인은 종목당 최근 2,000 이벤트만 메모리 유지.
- 알 수 없는 태그/사유는 버리지 않고 `UNKNOWN`으로 집계하고 설정 화면에서 목록 확인 가능.
- 로그 원문은 앱 외부로 전송하지 않는다.

## 7. 테스트 픽스처와 수용 기준

- 서버에서 하루치 로그를 복사해 `fixtures/20261104/`에 둔다(`scp -r user@host:/data/srv/autobot/runtime/log/20261104 fixtures/`). 메인 로그 58,127줄,
  buy_sell 116줄, 종목 파일 35개가 들어 있다.
- `vitest` 단위 테스트: (a) 2.2 정규식으로 메인 로그 전 줄이 파싱되거나 "태그 없음"으로 분류된다(예외 0), (b) 3.1 표의 각 행에 대해 예제 한 줄 → 기대 `condFunc`,
  (c) 2.3 거래 블록 파싱: 2026-11-04 픽스처에서 **매도 12건, 합계 -20,265원** 이 나와야 한다, (d) 증분 읽기: 파일을 두 조각으로 나눠 먹여도 결과 동일,
  (e) 불완전 마지막 줄 보류.
- 수용 기준: 픽스처로 앱을 열었을 때 159010이 `_004 하드 손절`로 매도된 거래가 타임라인에 보이고, 다이어그램에서 `_014`가 오늘 가장 많이 막힌 노드(257회)로 표시된다.
  (2026-11-04 메인 로그 집계: `TRIPLE_CROSS_MISSING_*` 3,000+ 중 대부분이 개장 전 `TICK_ABSTAIN_NO_FILE`, `HYBRID_3MIN_CTX_LOW_ENTRY_ATR` 257, `SESSION_OPEN_WARMUP` 63.)

## 8. npm 스크립트

```
npm run dev         # electron-vite dev (핫리로드)
npm run test        # vitest
npm run build:win   # electron-builder --win nsis  -> dist/*.exe
```

## 9. 하지 말 것

- 서버 파일 쓰기/삭제, 주문 API 호출, r003 재시작 - 모두 금지(읽기 전용 SFTP + `python r015_export_conditions.py` exec만).
- 로그 포맷을 추측해서 "고치기" 금지 - 알 수 없는 줄은 UNKNOWN으로 보존.
- 서버 CPU를 쓰는 작업(grep/awk로 로그 전처리) 금지 - 원문을 받아 로컬에서 파싱한다(라즈베리파이는 매매 루프가 돌고 있다).

## 10. 먼저 물어볼 것 (구현 전 1회)

SSH 접속 정보(호스트/포트/사용자/인증 방식), 원격 루트 경로가 `/data/srv/autobot`인지, 앱 데이터 저장 위치, 과거 날짜 보관 기간.
