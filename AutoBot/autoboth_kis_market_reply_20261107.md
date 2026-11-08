# 회신: 임의 종목 시세 조회 (KIS, 읽기 전용) - `quote` 명령 + `bars` KIS 폴백 (요청 2026-11-06 → 처리 2026-11-07)

> 요청 문서 `autoboth/docs/server_kis_market_data_request_20261106.md`에 대한 서버 쪽 회신이다. 구현은
> `AutoBot/live_trading/r018_kis_market.py`(신규), `r016_control.py`(`quote` 명령), `r017_chart_data.py`(`bars` 폴백).
> 앱 쪽 호출 줄이기(재사용 창, 창 숨김 시 갱신 생략)는 같은 날 `autoboth/`에 반영했다(커밋 cb10b5b).

---

## 1. 요청과 달라진 점 (먼저 읽을 것)

| 항목 | 요청 | 구현 | 이유 |
|---|---|---|---|
| 토큰 | 당일 토큰 재사용, 없으면 `kis_auth.auth()` | **발급하지 않는다.** 유효한 토큰 파일이 없으면 rc 3 `kis_code: "token_not_ready"` | kis_auth의 토큰 파일 쓰기는 원자적이지 않다. 다른 프로세스가 동시에 `auth()`를 부르면 빈 파일을 읽고 재발급(1분 1회 제한)해서 r003 인증이 실패할 수 있다(2026-11-07 0바이트 토큰 파일 사례). 토큰은 r003이 시작할 때 발급한다 |
| 공용 호출 제한 | 초당 3건, 간격 0.25초 | **초당 2건, 간격 0.3초** (`runtime/cache/kis/.ratelimit` 파일 락) | r003은 이 제한을 모르고 같은 앱키로 초당 약 1.6건을 쓴다. 이 기능의 몫을 작게 둔다 |
| KIS 모듈 | `kis_auth`/`domestic_stock_functions` 사용 가정 | 쓰지 않는다. requests로 시세 TR 7개만 직접 GET | 주문 함수 import 자체를 없애고, pandas·websockets 기동 비용(1~2초)을 없앤다 |
| 체결 구분 | `buy`/`sell` | 항상 `""` | `inquire_ccnl` 응답에 매수·매도 구분 필드가 없다. 체결 강도(`strength`)는 준다 |
| `name` | 모든 `bars` 응답 | **KIS에서 받은 응답에만** | 파일 기반 응답의 종목명은 앱이 이미 파일 이름에서 얻는다 |
| 일봉 저장 | `daily_history`에 upsert해도 좋음 | 별도 캐시 `runtime/cache/kis/daily/<code>.json` | g025 재구성과 섞이지 않게 분리 |
| 인자 오류 | - | **이제 모든 인자 오류도 stdout JSON 하나 + rc 2** | 예: `{"ok": false, "error": "r016_control.py: argument cmd: invalid choice: 'quote' ..."}`. 앱의 `invalid choice` 검사는 `error` 문자열에 그대로 남아 있다. `--help`만 텍스트 |

## 2. 새 명령 `quote`

```
r016_control.py quote --code CODE [--trades N]     # N 최대 30
```
- 내부 호출: `inquire_price` + `inquire_asking_price_exp_ccn` + `inquire_ccnl` (+ 처음 보는 종목은 `search_stock_info` 1회, 이후 캐시).
- 같은 종목 2초 안 재요청은 캐시(`cached_ms` > 0).
- 출력은 요청서 3장의 형식 그대로다. 추가 키:
  - `partial_errors`: 호가·체결 중 실패한 것(현재가는 있음). 예: `["orderbook EGW00201: ..."]`. 이때 `orderbook`이 null이거나 `trades`가 빈 배열일 수 있다.
  - `trades[i][3]`은 항상 `""`.
- rc: 0 성공, 2 잘못된 코드(`invalid code 'ABC'`, `unknown code 999999`), 3 KIS 오류(`kis_code`: KIS `msg_cd` 또는 `token_not_ready` / `deadline` / `rate_busy` / `rate_storage` / `network` / `config`).

## 3. `bars`의 KIS 폴백 (파일이 있으면 기존과 완전히 같다)

| kind | 서버 파일 없을 때 | 캐시 |
|---|---|---|
| `day` / `week` / `month` | `inquire_daily_itemchartprice` 최근 3년(100행씩 최대 12회) | 장중 10분, 18:00 이후 받은 것은 그날 끝까지. 시간 제한으로 3년을 다 못 받으면 받은 만큼 주고 다음 요청 때 다시 받는다 |
| `min` 당일 | `inquire_time_itemchartprice` (30봉씩, 장 시작까지) | 5초. 이후는 마지막 미완성 분부터 1회 호출로 이어 받는다 |
| `min` 지난 날짜 | `inquire_time_dailychartprice` | 장 시작까지 다 받으면 영구 |
| `min --warmup K` | 직전 **거래일**(r014 달력) 분봉, 최대 3거래일 시도 | 위와 같음 |
| `tick` | 없음 | rc 2 `"tick bars need the tick collector (not subscribed): use min"` |

- `source`: `"kis:inquire_daily_itemchartprice"` 등 TR 이름. 응답 열·의미는 기존과 같다(`t` = 봉 시작 KST, `v` = 주).
- 세션 시각은 r014 달력을 따른다(수능일 10:00~16:30 등). 휴장일과 장 시작 전의 당일 분봉은 "데이터 없음"(rc 2).
- 요청 날짜 이전 일봉이 하나도 없으면(신규 상장 전 날짜 등) rc 2.
- 명령 하나의 모든 KIS 호출(본 요청 + 예열)이 **25초 예산**을 나눠 쓴다. 넘으면 받은 만큼 주거나 rc 3 `deadline`. 앱 타임아웃 30초 안에 끝난다.

## 4. 앱이 지켜 줄 것 (이미 반영함, cb10b5b)

- `quote`: 같은 종목은 장중 8초, 장외 5분 재사용. 오류는 20초 재사용. 같은 종목 동시 요청은 한 번으로 합친다.
- `bars --kind day`: 10분 재사용, 실패 1분. 장외 분봉 5분.
- 창이 최소화되거나 숨겨지면 차트 주기 갱신을 하지 않는다(종목·봉 종류 변경은 그대로 로드).
- `token_not_ready`면 "봇(r003)이 시작되면 토큰이 생깁니다" 같은 안내를 보여 주면 좋다(앱에 반영함).

## 5. 확인 상태

| 회신 요청 항목 | 상태 |
|---|---|
| 1. `quote` 정상 출력(장중) / 오류 출력 | 오류 출력은 확인함(`token_not_ready` rc 3, `invalid code` rc 2). **정상 출력은 월요일 장중**에 r003 토큰으로 확인 예정 |
| 2. `bars` KIS 폴백 출력 | 모킹 테스트로 형식 확인(일봉 3년 페이징, 당일·과거 분봉, 예열, tick 오류). 실데이터는 월요일 |
| 3. 공용 제한 + r003 동시 10분 관찰(EGW00201) | **월요일 장중 예정** |
| 4. 토큰 재사용(새 발급 없음) | 코드상 발급 경로 없음(테스트로 확인). 월요일에 토큰 파일 시각으로 재확인 |

테스트: `AutoBot/tests/test_r018_kis_market.py` 33개(HTTP 모킹), 서버 전체 테스트 통과. Codex 검토 2회(1차 7건, 2차 3건 + 위험 3건 반영). r016은 quote/bars에 절대 시각 27초 강제 중단(SIGALRM)을 둬서 느린 응답에서도 JSON을 낸다.
