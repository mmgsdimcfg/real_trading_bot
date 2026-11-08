# real_trading_bot

한국투자증권 Open API 기반 국내주식 자동매매.

| 폴더 | 내용 | 실행 위치 |
|---|---|---|
| `autobot/` | 자동매매 서버 코드 (r003 실매매, 스캐너/수집기/백테스트, 제어 CLI r016) | 라즈베리파이 Ubuntu |
| `AutoBotH/` | AutoBotH 윈도우 앱 (Electron, SSH로 로그 표시 + r016 호출로 제어) | Windows 11 |

- 매매는 서버에서만 실행한다. 앱은 표시와 요청(`autobot/live_trading/r016_control.py`)만 한다.
- 서버 배치: `/data/srv/autobot/{real_trading_bot, runtime, open-trading-api}`.
  `runtime/`(데이터·로그·설정 오버라이드)와 `open-trading-api/`는 저장소 밖, 저장소 폴더와 같은 위치에 둔다
  (`AUTO_TRADING_RUNTIME`, `OPEN_TRADING_API_ROOT` 환경변수로 바꿀 수 있다).
- 서버-앱 연동 문서: `autobot/autoboth_*.md`, `autobot/condition_viewer_implementation_prompt_20261104.md`.
- 커밋 메시지 앞에 `autobot:` / `autoboth:`를 붙여 구분한다.
