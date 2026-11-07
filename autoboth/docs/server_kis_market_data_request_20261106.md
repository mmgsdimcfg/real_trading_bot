# 서버 요청 프롬프트: 임의 종목 시세 조회 (KIS OpenAPI, 읽기 전용) - AutoBotH 차트 탭 (2026-11-06)

> 서버 저장소(`auto_trading_src`) 담당 코딩 에이전트(또는 개발자)에게 그대로 주는 프롬프트다.
> 어제 요청(`server_data_request_prompt_20261106.md`)에 대한 회신 `autoboth_chart_data_reply_20261106.md`로 `bars` / `indicators` / `daily_history` /
> `status.collector`가 생겼고, AutoBotH v2.2.0_008부터 이 명령들을 쓴다. 이번 요청은 **서버에 수집 파일이 없는 종목과 시간대**를 KIS OpenAPI 시세 조회로 채우는 것이다.

---

## 0. 배경 (실제 확인한 것)

AutoBotH 차트 탭은 6자리 종목 코드를 직접 입력해 차트를 본다. 2026-11-06 09:30에 확인한 결과:

| 요청 | 지금 결과 |
|---|---|
| `bars --date 20261105 --code 005930 --kind day` | rc 2 `no daily data for 005930 around 20261105` - 삼성전자·SK하이닉스(000660) 등은 날짜 폴더(약 3,290종목)와 `daily_history`(3,045종목)에 없다 |
| `bars --date 20261106 --code 035720 --kind min` (장중) | rc 2 `no tick file / _10s file` - 오늘 `_10s`는 장 마감 후 g001이 만들고, 틱은 감시·후보 40종목뿐 |
| 현재가·호가·체결 | 명령 없음 (`quote`는 invalid choice) |

## 1. 지켜야 할 것 (반드시)

1. **읽기 전용 시세 TR만** 쓴다: 현재가(`inquire_price`), 호가(`inquire_asking_price_exp_ccn`), 체결(`inquire_ccnl`), 당일 분봉(`inquire_time_itemchartprice`),
   일별 분봉(`inquire_time_dailychartprice`), 기간별 시세(`inquire_daily_itemchartprice`), 종목 정보(`search_stock_info`). 주문·잔고·계좌 TR은 import도 하지 않는다.
2. **r003 매매에 영향이 없어야 한다.** KIS REST 호출 한도(실전 앱키당 초당 건수)는 r003과 같은 앱키를 나눠 쓴다.
   - 이 기능 전체에 **프로세스 간 공용 제한**을 둔다: 예) `runtime/cache/kis/.ratelimit` 파일 락 + 최근 호출 시각으로 **초당 최대 3건**, 호출 사이 최소 0.25초.
   - `EGW00201`(초당 거래건수 초과)을 받으면 재시도하지 말고(최대 1회, 1초 뒤) 캐시가 있으면 캐시를, 없으면 rc 3으로 돌려준다. 앱은 다음 폴링에 다시 묻는다.
   - r003이 쓰는 토큰을 **새로 발급하지 않는다.** `kis_auth`의 당일 토큰 파일(`~/KIS/config/KIS<날짜>`)을 읽어 재사용하고, 토큰이 없거나 만료됐을 때만 `kis_auth.auth()`의 기존 흐름을 따른다.
     토큰 값·앱키는 출력·로그·캐시 파일 어디에도 쓰지 않는다.
3. **캐시로 호출 수를 줄인다** (`runtime/cache/kis/`):
   - 현재가/호가/체결: 같은 종목 2초 안 재요청은 캐시 응답 (`cached_ms` 표시).
   - 당일 분봉: 이미 받은 분은 다시 받지 않는다(마지막 미완성 분만 갱신). 지난 날짜 분봉·일봉은 날짜·종목별로 한 번 받으면 끝(장 마감 후 확정).
   - 일/주/월봉: 하루 한 번(장중엔 당일 행만 현재가로 갱신).
4. 출력·종료 코드는 `r016_control.py` 규칙 그대로: stdout JSON 하나, rc 0 성공 / 2 입력·데이터 없음 / 3 KIS 오류(`kis_code`, `error`에 KIS 메시지).
   한 호출 30초 안에 끝낼 것(당일 분봉 처음 채우기 13회 호출 ≈ 4~5초).
5. 코드 검사: `^[0-9A-Z]{6}$` 외는 rc 2. 시장 구분은 `J`(KRX) 우선, 결과가 비면 `UN`(통합)/`NX`는 쓰지 않는다(틱·봇과 같은 KRX 기준).

## 2. 요청 A - `bars`에 KIS 폴백 추가 (기존 출력 형식 그대로)

```
r016_control.py bars --date YYYYMMDD --code CODE --kind tick|min|day|week|month [--ticks T] [--interval M] [--warmup K] [--since-ts ISO]
```
지금처럼 **파일(틱 → `_10s` → `daily_history`/`_daily.csv`)을 먼저** 쓰고, 파일이 없을 때만 KIS로 채운다.

| kind | 파일 없을 때 KIS | 비고 |
|---|---|---|
| `min` 당일 | `inquire_time_itemchartprice` (30봉/회, 09:00부터 현재까지) → M분 집계 | `--since-ts` 증분이면 마지막 1~2회만 호출 |
| `min` 지난 날짜 | `inquire_time_dailychartprice` (실전 전용, 120봉/회) | KIS가 주는 기간까지만. 없으면 rc 2 |
| `day/week/month` | `inquire_daily_itemchartprice` (D/W/M, 100행/회) **최근 3년** | 받은 일봉은 `daily_history/<code>.csv`에 `source_date` 규칙대로 upsert해도 좋다 |
| `tick` | 없음 (REST에 틱 조회 없음) | rc 2 `error`: `"tick bars need the tick collector (not subscribed): use min"` 그대로 |

- `--warmup K`: min이면 직전 거래일 분봉을 같은 방식(파일 → KIS)으로 채운다.
- 출력은 지금 `bars`와 **같은 키·같은 `columns`(`t,o,h,l,c,v`)**, `t` = 봉 시작(KST ISO). 추가 키:
  - `"source": "kis:inquire_time_itemchartprice"`처럼 TR 이름을 넣는다(파일이면 지금처럼 경로).
  - `"name": "삼성전자"` (모든 `bars` 응답에 넣어 주면 앱이 종목명 표시에 쓴다. `search_stock_info` 결과를 `runtime/cache/kis/names.json`에 캐시).
  - 거래량 단위는 **주**(10초 파일 ÷6 보정과 같은 기준).
- 오류 예: `bars --code 999999 --kind day` → rc 2 `{"ok": false, "error": "unknown code 999999"}` (KIS가 빈 결과).

## 3. 요청 B - 새 명령 `quote` (현재가 + 호가 10단계 + 최근 체결)

```
r016_control.py quote --code CODE [--trades 30]
```
`inquire_price` + `inquire_asking_price_exp_ccn` + `inquire_ccnl` 3회(캐시 2초). 장 마감 후에도 마지막 값이 오면 그대로 준다.
```json
{"ok": true, "code": "005930", "name": "삼성전자", "market": "KOSPI", "asof": "2026-11-06T09:31:02", "source": "kis", "cached_ms": 0,
 "price": 71200, "change": 800, "change_pct": 1.14, "prev_close": 70400, "base_price": 70400,
 "open": 70600, "high": 71500, "low": 70500, "volume": 8123456, "amount": 578000000000,
 "upper_limit": 91500, "lower_limit": 49300, "w52_high": 88800, "w52_high_date": "20260712", "w52_low": 49900, "w52_low_date": "20251203",
 "per": 13.2, "pbr": 1.21, "market_cap_eok": 4250000, "vi": false, "halted": false,
 "strength": 103.2,
 "orderbook": {"asof": "09:31:01", "asks": [[71300, 12000], [71400, 9000]], "bids": [[71200, 15000], [71100, 8000]],
               "total_ask": 210000, "total_bid": 190000},
 "trades": [["09:31:01", 71200, 15, "buy"], ["09:31:00", 71100, 3, "sell"]]}
```
- `asks`: 매도호가 1→10 (낮은 가격부터), `bids`: 매수호가 1→10 (높은 가격부터), `[가격, 잔량]`.
- `trades`: 최근 체결 최대 `--trades`개, 최신이 먼저. 4번째 값은 체결 구분(`buy` 매수체결 / `sell` 매도체결 / `""` 모름).
- `strength`: KIS 체결강도(`tday_rltv`). `vi`/`halted`: VI 발동·거래정지 여부(현재가 응답의 해당 필드, 없으면 false).
- 오류: rc 2 잘못된 코드 / rc 3 KIS 오류 `{"ok": false, "error": "...", "kis_code": "EGW00201"}`.

## 4. 앱이 이렇게 쓴다 (호출량 산정용)

- 차트 탭을 열어 둔 동안, 오늘 날짜면 **10초마다** `quote` 1회 + `bars --since-ts` 1회. 종목을 바꾸거나 봉 종류를 바꿀 때 `bars` 1회(처음 채우기).
- 지난 날짜·일봉은 폴링하지 않는다. 앱은 `r016_control.py` 호출을 직렬 큐 하나로 보낸다(동시 2건 없음).
- 예상: 차트 탭 1개당 평균 **0.3~0.5 TR/초** - 위 공용 제한(초당 3건) 안에서 r003 몫을 거의 쓰지 않는다.

## 5. 회신 요청

1. `quote` 정상 출력 1개(장중), 오류 출력 1개.
2. `bars` KIS 폴백 출력: `--kind min`(당일, 비수집 종목), `--kind day`(005930), `--kind tick` 오류.
3. 공용 호출 제한 방식과, r003과 동시에 돌렸을 때 `EGW00201`이 생기는지 장중 10분 관찰 결과.
4. 토큰 재사용 확인(새 발급이 일어나지 않음 - `kis_auth` 로그 또는 토큰 파일 시각).
