# real_trading_bot

한국투자증권 Open API 기반 국내주식 자동매매.

| 폴더 | 내용 | 실행 위치 |
|---|---|---|
| `AutoBot/` | 자동매매 코드 (r003 실매매, 스캐너/수집기/백테스트, 제어 CLI r016, 데이터 작업 r021) | 윈도우 노트북(메인) / 라즈베리파이 Ubuntu(예비) |
| `AutoBotH/` | AutoBotH 윈도우 앱 (Electron). 로컬 모드: 노트북에서 r003 실행·표시·제어, 원격 모드: SSH로 라즈베리파이 표시·제어 | Windows 11 |

- 같은 `AutoBot/` 코드가 두 호스트에서 돈다. 계좌는 하나이므로 r003은 한 번에 한 호스트에서만 매매한다
  (`runtime/config/host_role.json` = primary / standby, standby 호스트는 r003 시작을 거부).
- 배치 (호스트마다 저장소 폴더와 같은 위치에 `runtime/`, `open-trading-api/`):
  - 라즈베리파이: `/data/srv/autobot/{real_trading_bot, runtime, open-trading-api}`, Python `~/venv`, systemd `autobot-r003.service`
  - 노트북: 예) `D:\make_utils\{real_trading_bot, runtime, open-trading-api, venv}`, Python 3.12 venv, `live_trading/r020_supervisor.py`
  - `AUTO_TRADING_RUNTIME`, `OPEN_TRADING_API_ROOT` 환경변수로 바꿀 수 있다.
- 운영 문서: `AutoBot/autoboth_local_mode_20261110.md` (노트북 메인 + 라즈베리파이 예비 구조, 전환 절차, r016 새 명령).
  이전 서버-앱 연동 문서: `AutoBot/autoboth_*.md`, `AutoBot/condition_viewer_implementation_prompt_20261104.md`.
- 커밋 메시지 앞에 `autobot:` / `autoboth:`를 붙여 구분한다.
