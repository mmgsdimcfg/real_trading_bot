<!--
Update log
- [2026-11-20] type=docs - v1.0.0_001: new AutoBotHtsH (HtsH + AutoBotH in one app)
- [2026-11-21] type=docs - v1.0.5_001: 바닥 반등 초입 감시 (TEST 탭, 팝업 알림, 설정, 리플레이)
-->
# AutoBotHtsH

HtsH(한국투자증권 OpenAPI 조회 전용 개인 HTS)의 화면에 AutoBotH(r003 자동매매 봇 뷰어·원격 제어)의 기능을 합친 Windows 앱입니다.

## 화면

- HtsH와 같은 **여러 창(MDI) 작업공간**입니다. 창을 끌어 옮기고, 크기를 바꾸고, 최소화·최대화·정렬할 수 있습니다.
- **HTS 창** (툴바 `+ 차트`, `+ 현재가` …): 차트, 현재가, 종목정보, 관심종목, 순위, 지수, 뉴스. KIS 앱키로 로그인하거나 데모 모드로 씁니다.
- **봇 창** (툴바 `봇 ▾`): AutoBotH의 탭 11개가 창 하나씩입니다. 같은 종류는 하나만 열리고, 다시 고르면 열린 창으로 이동합니다.
  - 운영: 제어, 데이터 작업, 수집기, 파이프라인
  - 분석: 종목 보드, 타임라인, **봇 차트**, 보유 현황, 매매 결과, 가상 매매
  - 시스템: **봇 설정·연결**
- 이름이 겹치는 기능은 봇 쪽에 `봇` 을 붙였습니다(봇 차트 = AutoBotH 차트 탭, 봇 설정·연결 = AutoBotH 설정·연결 탭).
- 툴바 아래 한 줄은 AutoBotH 상태 줄입니다(연결 · 봇 실행 상태 · 날짜 · 수신/경고/시장/감시/조건).

## 설정과 데이터

- 앱 폴더: `%APPDATA%\autobothtsh` (HtsH / AutoBotH와 따로).
- 처음 실행할 때 가져오는 것:
  - AutoBotH 연결 설정. SSH 비밀번호와 프록시 비밀번호는 앱마다 암호화 키가 달라 가져오지 않으니 **봇 설정·연결**에서 다시 입력하세요.
  - HtsH 관심종목(`watchlist\watchlist.json`).
- KIS 앱키는 HtsH와 따로 저장됩니다. 처음에 한 번 로그인하세요.
- 관심종목 picks 파일(`_YYYYMMDD_picks.txt`)은 `%APPDATA%\autobothtsh\watchlist\` 에 두면 MMDD 탭으로 등록됩니다.

## 바닥 반등 초입 감시

관심종목 창에서 이름이 **TEST** 인 탭의 종목을 1분봉으로 감시하다가, 급락 뒤 반등 초입 조건이 맞으면 화면 우하단에 항상 위 팝업을 띄웁니다. 감지와 알림만 하고 주문은 내지 않습니다.

- **쓰는 법**: KIS 앱키로 로그인하고 관심종목 창에 `TEST` 탭을 만들어 종목을 넣으면 됩니다(탭 더블클릭 = 이름 바꾸기). 탭은 5분마다 다시 읽습니다. 데모 모드에서는 감시하지 않습니다.
- **설정**: ☰ > **반등 감시 설정**. 감시 켜기/끄기, 감시할 탭(고르기·만들기), 그 탭의 종목 추가·삭제, 아래 조건의 모든 값, 운영 시간, 알람 테스트, 설정·기록 폴더 열기를 한 창에서 합니다. 저장하면 다음 분부터 적용됩니다.
- **상태**: 제목 줄의 `반등 감시 TEST · N종목` (표시만). 마우스를 올리면 마지막 순회 시각과 오류가 보입니다.
- **동작**: 매분 2초에 탭의 종목을 차례로 조회해(첫 조회 뒤에는 종목당 1회 호출) **완성된 봉**만 판정합니다. 같은 종목은 30분 동안 다시 알리지 않습니다. 팝업의 `열기` 는 그 종목을 앱의 창들에 띄웁니다.
- **운영 시간**: 평일 KRX 09:00~15:30, NXT 프리 08:00~08:50, NXT 애프터 15:40~20:00. NXT 시간에는 NXT 거래 종목만 조회합니다.
- **파일** (`%APPDATA%\autobothtsh\rebound\`, 설정 창의 `설정·기록 폴더` 버튼):
  - `config.json`: 설정 창이 쓰는 파일. 직접 고쳐도 다음 분부터 반영됩니다. 지우면 기본값으로 다시 만들어집니다.
  - `detections_YYYYMMDD.csv`: 감지 내역 전체(시각, 종목, 지표값, 점수, 알림 여부).
- **모의투자 접속**은 지난 날짜 분봉을 받을 수 없어, 당일 봉이 200개 쌓인 뒤부터 판정합니다.

### 감지 조건

필수 조건 6개를 모두 충족하고 가점이 `bonus.minScore`(3점) 이상이면 감지입니다. 돌파는 급락 봉 뒤, 최근 `lookback`(25)봉 안에 있어야 합니다.

| | 필수 조건 | 설정 항목 |
|---|---|---|
| A | 최근 40봉 안에 급락 봉: 전봉 대비 -0.8% 이하 또는 아랫꼬리가 몸통의 2배 이상, 거래량이 앞 60봉 평균의 3배 이상, 그 뒤 저가가 더 낮아지지 않음. 급락 봉까지의 낙폭(앞 60봉 최고가 -> 급락 봉 저가)이 8%를 넘으면 악재성 급락으로 보고 제외 | `plunge.*` (`fallWindow`, `maxFallPct`, 0 = 제한 없음) |
| B | RSI가 시그널을 상향 돌파, 최근 5봉 중 3봉 이상 시그널 위, 현재 봉도 시그널 위, 현재 RSI 45~68 | `recent`, `recentMin`, `requireCurrentAbove`, `rsi.*` |
| C | MACD가 0선 아래에서 시그널을 상향 돌파한 뒤 계속 위, 오실레이터 3봉 연속 확대 | `macd.oscRising` |
| D | DI+가 DI-를 상향 돌파, 최근 5봉 중 3봉 이상 위, 현재 봉도 위 | `recent`, `recentMin`, `requireCurrentAbove` |
| E | MA5가 MA20을 골든크로스, 종가 > MA5 > MA20 | |
| F | MA5 각도 14° 이상. 각도 = atan(1봉당 % 변화 × `k`), 3봉 전 MA5와 비교 | `angle.*` |

**장 경계 제외** (`edge.openMin`, `edge.closeMin`, 기본 15분): 각 운영 시간대의 처음 15분과 마지막 15분(KRX 09:00~09:14, 15:16~15:30, NXT 프리·애프터도 같은 방식)에는 감지하지 않고, 그 시간의 봉은 급락 봉으로도 치지 않습니다. 개장·마감 때는 거래량이 원래 몰려 "평균의 3배" 조건이 저절로 충족되기 때문입니다. 0으로 두면 끕니다.

가점(각 1점): ADX가 내려왔다가 최근 4봉 안에 바닥을 찍고 반등, 종가가 MA60/MA120을 돌파했거나 0.2% 이내, 볼린저 폭이 급락 때보다 좁고 종가가 중심선 이상, 전환선 > 기준선이고 종가가 구름 안이나 위, MACD가 0선 근처(가격의 0.05% 이내)이거나 최근 3봉 안에 0선 돌파, 반등 구간 평균 거래량이 급락 봉보다 적음. VWAP 위치는 조건이 아니고 팝업에 표시만 합니다.

지표는 차트와 같은 코드(`src/shared/indicators.ts`)로 계산합니다. 기본값은 차트 기본 설정과 같습니다(RSI 14 / 시그널 6, MACD 12·26·9, DMI 14, 볼린저 20·2, 일목 9·26·52, 계산 방식 `mts`). 차트 설정을 바꿨다면 `config.json` 의 `indicator` 도 같이 바꾸세요. `rsiMethod` / `adxMethod` 를 `wilder` 로 두면 Wilder 방식입니다.

### 리플레이

저장된 1분봉(CSV `date,time,open,high,low,close,volume` 또는 앱의 차트 캐시 `chartcache\real_<코드>_min_.json`)을 봉 하나씩 흘려보내 감지 시점을 출력합니다.

```
npm run replay -- tests/rebound_448900_20261120.csv --from 13:00 --to 15:20
npm run replay -- <파일> --date 20261120 --explain 14:55      # 그 봉의 조건별 판정
npm run replay -- <파일> --all                                # 필수 조건만 충족한 봉도 표시
npm run replay -- <파일> --config <config.json> --set angle.min=15 --set recentMin=4
```

## AutoBotH와 함께 쓸 때

- 봇 실행(r003), 감시 프로세스(r020), 데이터 작업(r021), 틱 수집기(g011)는 서버 쪽 잠금 파일을 쓰므로 어느 앱에서 시작해도 한 번만 실행됩니다.
- 이 PC(로컬 모드) 스케줄러(틱 수집기 확인 / 장 마감 작업 / 라즈베리파이 감시목록 복사)는 두 앱 중 **하나만** 돌립니다.
  - 공유 파일: `%APPDATA%\autoboth-shared\scheduler.json`
  - 다른 앱은 대기하고, 스케줄러 메시지에 어느 앱이 실행 중인지 표시합니다.
  - 마지막 장 마감 작업 날짜도 공유해서 같은 날 두 번 실행하지 않습니다.

## 소스 구성

| 위치 | 내용 |
|---|---|
| `src/main`, `src/preload`, `src/renderer`, `src/shared` | HtsH (v0.1.5_002 기준) + 통합 코드 |
| `src/bot/{main,preload,shared,renderer/src}` | AutoBotH (v2.3.32_003 기준) 사본. 상대 경로 구조를 그대로 둠 |
| `src/main/importOnce.ts` | 첫 실행 때 설정 가져오기 |
| `src/shared/rebound.ts` | 바닥 반등 감지(순수 함수)와 설정 기본값 |
| `src/shared/reboundReplay.ts`, `scripts/rebound-replay.cjs` | 리플레이 |
| `src/main/reboundPoller.ts`, `src/main/reboundAlert.ts` | 매분 순회, 팝업·알림음·감지 기록 |
| `src/renderer/src/windows/BotWindow.tsx` | 봇 화면을 창으로 띄우는 부분, 상태 줄, 확인 창 |
| `src/bot/renderer/src/styles.scoped.css` | AutoBotH 스타일을 `.bot-root` 안으로 한정한 생성 파일 (`npm run scope-css`) |
| `tests/` (HtsH), `tests/bot/` (AutoBotH) | vitest |

AutoBotH나 HtsH를 고친 뒤 이 앱에도 반영하려면 해당 파일을 같은 위치로 복사합니다. AutoBotH의 `styles.css`를 바꿨다면 `npm run scope-css` 를 실행합니다. 통합 때 바꾼 파일은 각 파일 맨 위 Update log의 `AutoBotHtsH` 항목에 적혀 있습니다.

## 명령

```
npm install
npm run dev        # 개발 실행
npm start          # 빌드 후 실행
npm test           # vitest
npm run replay -- <분봉 파일>   # 바닥 반등 감지 리플레이
npm run typecheck
npm run dist       # 설치 파일 (dist/)
```

화면 시험: `AUTOBOTHTSH_USERDATA=<빈 폴더>` 로 실행하면 그 폴더를 앱 폴더로 쓰고 아무 설정도 가져오지 않습니다(봇 연결·스케줄러 없음). `HTSH_DEMO=1` 은 데모 모드로 시작합니다.
