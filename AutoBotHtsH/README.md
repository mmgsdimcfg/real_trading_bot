<!--
Update log
- [2026-11-20] type=docs - v1.0.0_001: new AutoBotHtsH (HtsH + AutoBotH in one app)
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
npm run typecheck
npm run dist       # 설치 파일 (dist/)
```

화면 시험: `AUTOBOTHTSH_USERDATA=<빈 폴더>` 로 실행하면 그 폴더를 앱 폴더로 쓰고 아무 설정도 가져오지 않습니다(봇 연결·스케줄러 없음). `HTSH_DEMO=1` 은 데모 모드로 시작합니다.
