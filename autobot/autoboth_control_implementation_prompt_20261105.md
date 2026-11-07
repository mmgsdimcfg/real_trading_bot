# 구현 가이드 프롬프트: AutoBotH 원격 제어 기능 (r003 실행/정지, 파이프라인 설정 편집, 감시목록 편집)

> 이 문서는 그대로 코딩 에이전트(또는 개발자)에게 주는 프롬프트다. 기존 AutoBotH(조건 파이프라인 뷰어, Electron + React,
> `ssh2`로 서버 로그를 읽는 읽기 전용 앱)에 **제어 기능**을 추가한다. 서버 쪽 준비물은 저장소에 이미 있다:
> `live_trading/r016_control.py`(제어 CLI), `live_trading/r003_service.sh` + `live_trading/systemd/autobot-r003.service`(systemd user 서비스),
> `r001_define_config.py`의 라이브 오버라이드 로더. 이전 가이드 `condition_viewer_implementation_prompt_20261104.md`의 로그 문법·SSH 설정은 그대로 유효하다.

---

## 0. 당신의 임무

AutoBotH에 다음 4가지를 추가한다. 모든 쓰기 동작은 **서버의 `r016_control.py`를 SSH exec으로 호출**해서 수행한다. 앱은 서버 파일을 직접
SFTP로 쓰지 않고, 주문 API를 절대 호출하지 않으며, 검증 로직을 앱 안에 복제하지 않는다(서버가 거부하면 그 메시지를 그대로 보여준다).

1. **제어 탭**: 봇(r003) 실행/정지/재시작, 상태 표시(실행 중, PID, 시작 시각, dry-run 여부, 리스크 프로파일, 재시작 횟수), "미적용 변경 있음" 배지.
2. **파이프라인 노드 설정 편집**: 노드를 눌렀을 때 오른쪽 패널의 "설정값" 표를 편집 가능하게 만든다. 저장하면 서버 오버라이드 파일에 기록되고,
   r003을 재시작해야 적용된다(배지 표시).
3. **파이프라인 켜기/끄기**: 각 노드의 `ENABLE_*` 상수를 토글로 노출한다(2번과 같은 경로, 값만 bool).
4. **감시목록(r004) 편집**: 종목 코드 목록을 보고 고치고 저장한다(서버가 백업 후 교체). 역시 재시작 후 반영.

사용자는 윈도우 PC에서 앱만 열고 봇을 켜고 끄며, 오늘 막힌 조건의 설정값을 고치고 재시작하는 흐름을 원한다.

## 1. 서버 계약 (전부 `r016_control.py`)

호출 형식(기존 r015 호출과 동일한 방식):
```
cd /data/srv/autobot/real_trading_bot/autobot && R016_BY=autoboth ~/venv/bin/python live_trading/r016_control.py <command> [args]
```
- 출력: **항상 stdout에 JSON 하나**. stderr는 무시하되 로그에 남긴다.
- 종료 코드: `0` 성공, `2` 검증 실패/거부(JSON에 `error` 또는 `rejected`), `3` systemctl 실패(JSON `ok:false`, `stderr`).
- `--by NAME`(또는 env `R016_BY`)은 변경 이력에 남는 작성자. 앱은 `autoboth@<윈도우 사용자명>` 형태로 보낸다.
- 인자에 공백/한글이 들어가면 **반드시 셸 인용**한다(`--codes "005930,삼성전자;000660,SK하이닉스"`). 따옴표·세미콜론·백틱이 든 값은 앱에서 먼저 거부한다.
- 모든 호출에 타임아웃 30초. 동시에 두 개의 쓰기 명령을 보내지 않는다(앱 안에 직렬 큐 하나).

### 1.1 `status`
```json
{
  "generated_at": "2026-11-06T00:40:15",
  "service": {"available": true, "unit": "autobot-r003.service", "loaded": true, "active_state": "active",
              "sub_state": "running", "main_pid": 4242, "started_at": "2026-11-06T08:30:01", "restarts": 0, "result": "success"},
  "running": true,
  "external_r003": [],
  "applied_state_known": true,
  "snapshot": {"pid": 4250, "started_at": "2026-11-06T08:30:04", "dry_run": false, "env_dv": "real", "risk_profile": "neutral",
               "overrides_file": ".../runtime/config/live_overrides.json", "overrides_errors": [],
               "watchlist_file": ".../r004_trade_watchlist_today.txt", "watchlist_mtime": 1762385171.2, "watchlist_codes": 7,
               "trading_strategy_mode": "basic", "entry_trigger_mode": "triple"},
  "running_overrides": {"ENABLE_PROFIT_LOCK": false},
  "desired": {"dry_run": false, "risk_profile": "neutral", "env": {"R003_DRY_RUN": "0"}},
  "dry_run": false,
  "risk_profile": "neutral",
  "override_count": 2,
  "override_errors": [],
  "pending": {"applied_state_known": true,
              "overrides": {"added": ["MAX_OPEN_POSITIONS"], "changed": [], "removed": []},
              "env": {"dry_run": false, "risk_profile": false}, "watchlist": false, "rejected_in_file": {},
              "pending_keys": ["MAX_OPEN_POSITIONS"], "pending_restart": true},
  "pending_restart": true,
  "latest_main_log": "/data/srv/autobot/runtime/log/20261106/r003_trade_live_execute_20261106_083001.log",
  "watchlist_codes": 7,
  "collector": {"running": true, "pid": 1234, "pids": [1234], "subscribed": ["003490", "..."], "symbols": 23,
                "last_write": "2026-11-06T09:12:31", "stale": [{"code": "161390", "last_write": null}], "missing": [],
                "ticks_dir": ".../runtime/data/ticks/20261106"}
}
```
- `service.active_state`: `active`(실행 중) / `inactive` / `failed` / `activating`. `sub_state`가 `auto-restart`면 크래시 후 재시작 대기 중.
- `external_r003`: systemd 밖(screen/터미널)에서 돌고 있는 r003 프로세스 목록. **비어 있지 않으면 "실행"/"재시작" 버튼을 비활성화**하고
  경고를 띄운다(서버가 어차피 거부하지만, 사용자에게 이유를 먼저 보여준다).
- **`snapshot` = 지금 돌고 있는 r003이 실제로 쓰는 값**(r003이 시작 시 기록). `dry_run`/`risk_profile`/`running_overrides`는 여기서 온다.
  `desired`는 편집 파일의 값(다음 시작 때 적용될 값). 둘이 다르면 `pending`에 나타난다. **모드 표시는 반드시 `snapshot`(=최상위 `dry_run`) 기준**이고,
  `desired.dry_run`은 "다음 시작 모드"로만 표시한다.
- `applied_state_known:false`면 스냅샷이 없다(봇이 꺼져 있거나, 스냅샷을 쓰지 않는 옛 r003이 screen에서 돌고 있음). 이때 `dry_run`/`risk_profile`은
  null이고, `pending`은 "서비스 시작 이후 파일이 바뀌었나"로만 추정한다. UI에 "실행 상태 확인 불가(재시작 후 정확)" 문구를 보여준다.
- `pending.overrides`: `added`(새 오버라이드) / `changed`(값 변경) / `removed`(기본값으로 되돌림 - 역시 재시작 필요). `pending_keys`는 합집합.
- `override_errors`: 파일에 있지만 r001/r016이 거부하는 항목(사람이 읽는 문자열). 비어 있지 않으면 설정 패널 상단에 경고.
- `collector`: 틱 수집기 g011 상태. `stale`은 장중에만 채워진다(감시 종목 중 60초 이상 기록 없음). `missing`은 감시목록에 있는데 수집기가
  구독하지 않는 종목(감시목록을 바꾼 뒤 생김 → "수집기 재시작" 버튼 = `collector restart`).
- `status`는 10초 폴링(기존 로그 폴링 주기와 같은 타이머에 묶는다). 서비스 동작 직후에는 즉시 한 번 더 호출.

### 1.2 `start [--dry-run|--live] [--profile conservative|neutral|aggressive]` / `stop` / `restart [...]`
```json
{"ok": true, "verb": "start", "stderr": "", "service": {"active_state": "active", "main_pid": 4242, ...}}
{"ok": false, "error": "r003 already running outside the service (screen/terminal?) - stop it first", "external_r003": [{"pid": 777, "cmdline": "..."}]}
```
- 플래그를 주면 `<runtime>/config/r003.env`에 저장되고 다음 시작부터 계속 적용된다. 플래그 없이 `start`하면 이전 설정 유지.
- 서비스가 이미 `active`면 `start`는 거부된다(rc 3, `"service already active - use restart to apply changes"`) — 이때 env 파일도 바뀌지 않는다.
  `restart`도 외부 r003이 있으면 거부된다. r003 자체도 시작 시 `.r003.lock`을 잡아 두 번째 인스턴스는 스스로 종료한다(rc 3).
- `--dry-run`은 주문을 보내지 않는 모드. 제어 탭에 체크박스로 두고, **체크 해제 상태로 실행할 때는 "실주문 모드로 실행합니다" 확인 대화상자**를 띄운다.
- 정지는 r003이 상태를 저장하고 내려가도록 최대 90초 걸릴 수 있다. 버튼을 누른 뒤 `status`가 `inactive`가 될 때까지 2초 간격으로 확인하며 스피너를 보여준다.
- 장중(09:00~15:30 KST)에 `stop`/`restart`는 "보유 종목 매도 조건 평가가 중단됩니다" 확인 대화상자.

### 1.3 `schema [--all]`
편집 가능한 r001 상수 목록. 기본은 `ENABLE_*` + 현재 오버라이드된 키, `--all`이면 전체(약 380개).
```json
{"generated_at": "...", "overrides_file": "...", "applied_state_known": true, "errors": [],
 "keys": [{"key": "ENABLE_PROFIT_LOCK", "type": "bool", "base": true, "override": null, "has_override": false,
           "effective": true, "running": true, "pending": false}, ...]}
```
- `running`: 실행 중인 봇이 쓰는 값(스냅샷 기준; 스냅샷이 없으면 null). 표에는 `기본값 | 실행 중 값 | 오버라이드` 세 칸을 보여준다.
- `type`: `bool | int | float | str | list | dict | null`. 편집 위젯은 이 타입으로 고른다(bool=토글, int/float=숫자 입력, str=텍스트,
  list/dict=JSON 텍스트 영역 + 유효성 검사). list/dict는 원소 타입도 현재 값과 같아야 한다(문자열 코드 목록에 숫자를 넣으면 거부).
  숫자에 `NaN`/`Infinity`/`1e309`는 서버가 거부한다(앱에서도 유한수만 허용).
- `base`: 코드 + 리스크 프로파일이 정한 기본값. `override`: 앱/사용자가 저장한 값(없으면 null). `effective`: 지금 r001이 쓰는 값.
- `pending`: 오버라이드가 저장됐지만 실행 중인 봇에 아직 반영되지 않음.

### 1.4 `conditions`
`r015_export_conditions.py`와 같은 구조(`buy`/`sell` 배열, 각 항목에 `defines`·`values`)에 **`values_meta`**가 추가된다. 기존 r015 호출을 이것으로 교체하면 된다.
```json
"values_meta": {"ENABLE_MARKET_REGIME_GATE": {"editable": true, "type": "bool", "base": true, "override": null, "has_override": false, "pending": false},
                "MARKET_REGIME_ETF": {"editable": true, "type": "dict", "base": {"KOSPI": "069500", "KOSDAQ": "229200"}, ...}}
```
- `editable:false`인 키(경로·디렉터리 상수 등)는 읽기 전용으로 표시한다.
- 최상위에 `overrides`(현재 오버라이드 전체), `override_errors`, `overrides_file`도 온다.

### 1.5 `get KEY ...` / `set KEY=JSON ...` / `unset KEY ...` / `reset` / `history [-n N]`
```
set ENABLE_PROFIT_LOCK=false PROFIT_LOCK_GIVEBACK_PCT=0.006 EDGE_AI_MODE=shadow MARKET_REGIME_ETF='{"KOSPI":"069500"}'
-> {"ok": true, "accepted": {...}, "rejected": {}, "restart_required": true}
-> rc 2: {"ok": false, "accepted": {"MAX_OPEN_POSITIONS": 2}, "rejected": {"ENABLE_PROFIT_LOCK": "type mismatch (expected bool, got int; elements must match, no NaN/inf)"}, ...}
```
- `get` 응답의 각 값에도 `running`(실행 중 값)과 `pending`이 있다. `unset`으로 지운 키는 봇이 재시작할 때까지 `pending`이다(`pending.overrides.removed`).
- 값은 JSON으로 보낸다: `true/false`, `2`, `0.006`, `"gate"`(문자열은 따옴표 포함해 셸 인용), `[60,120]`, `{"a":"b"}`.
- 서버 규칙(앱이 복제하지 않는다, 알고만 있을 것): 존재하는 r001 대문자 상수만, bool↔int 교차 불가, int→float 가능, 경로/디렉터리/파일명/비밀 상수 불가.
- `rejected`가 있으면 해당 입력칸을 빨갛게 하고 서버 메시지를 그대로 보여준다. `accepted`는 이미 저장된 것이므로 UI를 갱신한다.
- `unset`은 기본값으로 되돌리기(행 단위 "기본값" 버튼), `reset`은 전부 되돌리기(확인 대화상자).
- `history`: `{"rows": [{"ts", "by", "action": "set|unset|reset|env|service.start|service.stop|service.restart|watchlist", "changes": {...}}]}`.
  제어 탭 하단에 최근 20건 표로 보여준다.

### 1.6 `watchlist get` / `watchlist set --codes "..."` / `watchlist set --stdin`
```json
{"path": "...r004_trade_watchlist_today.txt", "mtime": "2026-11-05T17:46:11", "header": ["# Daily live trading watchlist.", ...],
 "codes": [{"code": "161390", "name": "한국타이어앤테크놀로지"}, ...]}
{"ok": true, "count": 7, "backup": ".../runtime/config/watchlist_backup/r004_20261105_223000_123456.txt", "path": "...",
 "restart_required": true, "collector_missing": ["035720"],
 "collector_note": "running g011 does not subscribe these codes - run 'collector restart' ..."}
```
- `collector_missing`이 비어 있지 않으면 토스트에 "틱 수집기가 새 종목을 구독하지 않습니다 → [수집기 재시작]" 버튼을 붙인다.
- `--codes` 형식: `code,name;code,name;...` (name 생략 가능, 코드 6자리 숫자, 최대 100개, 중복은 첫 항목 유지). 긴 목록은 `--stdin`으로
  한 줄에 하나씩(`code,name`) 보낸다(ssh2 exec의 stdin에 쓰고 end).
- 서버는 기존 파일을 `watchlist_backup/`에 복사한 뒤 교체한다. 주석 헤더는 유지된다.
- 주의 문구를 UI에 고정 표시: "이 목록은 매일 아침 g002 스캐너 내보내기가 다시 씁니다. 수정은 당일 재시작까지만 유효합니다."
- 저장 후 `status.pending.watchlist.changed_since_start`가 true가 된다.

### 1.7 `collector status` / `collector restart`
- `collector status`: 1.1의 `collector` 블록과 같은 JSON.
- `collector restart`: 돌고 있는 g011에 SIGTERM → `{"ok": true, "signalled": [1234], "note": "..."}`. 수집기는 cron(g022, 장중 5분마다) 또는
  r003 감시자가 현재 r004 목록으로 다시 띄운다(최대 ~5분 공백). 수집기가 없으면 rc 3. 장중에 누르면 "최대 5분간 틱 수집이 끊깁니다" 확인.

### 1.8 차트 데이터 `indicators` / `bars` (2026-11-06 추가, 상세는 서버의 `live_trading/r017_chart_data.py` docstring과
`autoboth_chart_data_reply_20261106.md`)
```
r016_control.py indicators --date 20261105 --code 003490 --frame 3m [--last N]
r016_control.py bars --date 20261105 --code 003490 --kind tick --ticks 60 [--n N] [--warmup K] [--since-ts ISO] [--all-sessions]
r016_control.py bars --date 20261105 --code 003490 --kind min --interval 1 ...
r016_control.py bars --date 20261105 --code 003490 --kind day|week|month [--n N]
```
- 둘 다 읽기 전용, rc 0/2. 장중 10초 폴링은 `bars --since-ts <next_since>`로 한다(서버가 바이트 오프셋 캐시로 증분 파싱).
- `conditions` 의 각 조건에 `frame`("1m+tick", "3m", "live", "daily", "none")이 있어 노드에서 해당 차트를 바로 열 수 있다.

## 2. UI 명세

### 2.1 제어 탭 (새 탭 "제어", 파이프라인 탭 왼쪽 또는 설정/연결 옆)
```
┌ 봇 상태 ───────────────────────────────────────────────────────────┐
│ ● 실행 중  PID 4250  시작 2026-11-06 08:30:04  재시작 0회  모드: 실주문 │  ← snapshot 기준
│ 다음 시작: 프로파일 neutral ▾   [ ] dry-run(주문 안 냄)                │  ← desired (env 파일)
│ [ 실행 ]  [ 정지 ]  [ 재시작 ]        ⚠ 미적용 변경 2건 → 재시작 필요    │
│ 외부 r003 감지: 없음      틱 수집기: ● 23종목, 마지막 기록 09:12:31       │
├ 미적용 변경 ───────────────────────────────────────────────────────┤
│ 설정값 ENABLE_PROFIT_LOCK false (기본 true)      [기본값으로]            │
│ 감시목록 변경 22:30:00                                                │
├ 최근 변경 이력 (history -n 20) ──────────────────────────────────────┤
│ 22:30 autoboth@hjcho  set  ENABLE_PROFIT_LOCK true→false                │
└───────────────────────────────────────────────────────────────────┘
```
- 버튼 활성 규칙: 실행 = `!running && external_r003.length==0`; 정지/재시작 = `service.active_state=="active" && external_r003.length==0`.
- 모드 표시: `dry_run === null`이면 "확인 불가"(스냅샷 없음). `desired.dry_run !== dry_run`이면 "다음 시작부터 dry-run" 같은 보조 문구.
- 상단 상태바(기존 "연결됨 … 조건 서버")에 "봇: 실행 중/정지/미적용 변경" 요약을 추가한다.
- "외부 r003 감지"가 있으면 빨간 경고 + PID/명령줄 표시 + 안내: "서버의 screen 세션에서 r003을 종료한 뒤 앱에서 실행하세요."

### 2.2 노드 설정 편집 (기존 오른쪽 패널 확장)
- 노드 클릭 → 기존 "설정값" 표를 편집 모드로: 각 행 `키 | 기본값 | 현재 오버라이드 | 입력 위젯 | [기본값]`.
- `ENABLE_*` 키는 표 맨 위에 토글로, 노드 헤더에도 같은 토글을 미러링한다(파이프라인 켜기/끄기).
- 값 변경은 즉시 서버에 보내지 않는다. 패널 하단 `[저장] [취소]`, 저장 시 변경된 키만 모아서 `set` 한 번 호출. 결과의 `rejected`는 행별 에러로.
- 저장 성공 후: 해당 노드에 "미적용" 점(노란색) 표시, 제어 탭 배지 갱신, `conditions` 다시 불러오기.
- `editable:false`는 회색 읽기 전용. `type:dict|list`는 JSON 텍스트 영역, 앱에서 `JSON.parse` 실패 시 저장 전 막는다.
- 숫자 입력: `int`는 정수만, `float`는 소수 허용. 퍼센트 상수(`*_PCT`)는 코드가 소수 비율(0.004 = 0.4%)을 쓰므로 **값을 변환하지 말고 그대로 입력**받되
  옆에 `= 0.40%` 환산을 표시한다.
- 전체 상수 편집 화면(검색 가능, `schema --all`)을 설정 탭 안에 "고급: 전체 설정값"으로 둔다. 기본은 접혀 있다.

### 2.3 감시목록 편집 (종목 보드 탭에 "감시목록 편집" 버튼)
- `watchlist get`으로 표를 채우고 행 추가/삭제/순서 변경(순서 = r003의 활성 우선순위이므로 드래그 정렬 유지).
- 저장 → `watchlist set --stdin` → 결과 토스트 + 백업 경로 표시 + 제어 탭 배지.
- 코드 입력은 6자리 숫자 검사, 이름은 쉼표/세미콜론 금지(서버도 거부하지만 입력 단계에서 막는다).

## 3. 안전 규칙 (반드시)

1. 앱은 주문 API·KIS 토큰·계좌 정보에 접근하지 않는다. `~/KIS/config`는 읽지도 않는다.
2. 실주문 모드 실행, 장중 정지/재시작, `reset`, 감시목록 저장은 확인 대화상자를 거친다.
3. 서버 호출은 직렬 큐로 하나씩. 응답 전에는 같은 버튼을 다시 누를 수 없다.
4. 서버가 거부(rc 2/3)하면 앱은 재시도하지 않고 메시지를 보여준다. SSH 끊김으로 결과를 모르면 `status`로 다시 확인한다(`start` 후 실패처럼 보여도
   이미 켜졌을 수 있다).
5. 설정값은 재시작 전까지 봇에 반영되지 않는다는 사실을 UI에서 항상 보여준다(배지 + 패널 문구). 핫리로드는 없다.
6. 오버라이드는 저장소 밖 `<runtime>/config/`에만 쓰이고 git에는 들어가지 않는다. 앱이 저장소 파일을 수정하는 기능을 만들지 않는다.

## 4. 서버 쪽 사실 (구현 시 참고)

- r003은 시작할 때만 r001 상수와 r004 파일을 읽는다. 그래서 모든 변경은 "재시작 필요"다.
- 재시작하면 r003은 `live_state`(보유 종목 메타, 손절 종목, 재진입 횟수)를 저장했다가 복원한다. 틱 수집기(g011)는 재시작해도 죽지 않는다(`KillMode=process`).
  단, **미체결 주문의 추적(취소/정정/체결 확인)은 재시작하면 사라진다**(기존 수동 재시작과 같은 한계). 그래서 장중 정지/재시작 확인 대화상자에
  "미체결 주문이 있으면 수동 확인이 필요합니다"를 넣는다. 가능하면 장 시작 전/마감 후에 재시작하도록 안내.
- 감시목록을 바꾸고 재시작해도 이미 돌고 있는 g011은 구독 목록을 바꾸지 않는다 → `collector_missing`이 있으면 `collector restart`까지 해야 새 종목의
  틱 트리거가 동작한다.
- systemd 유닛은 부팅 시 자동 시작하지 않는다. 서버 재부팅 후에는 사용자가 앱에서 실행 버튼을 눌러야 한다(제어 탭에 안내 문구).
- 서비스 stdout은 `journalctl --user -u autobot-r003`에, 매매 로그는 기존처럼 `runtime/log/<date>/`에 쓰인다. 앱의 로그 파서는 변경 없음.
- g003 백테스트도 같은 오버라이드 파일을 읽는다(라이브 parity). 백테스트에서 끄려면 서버 env `AUTO_TRADING_LIVE_OVERRIDES=0`. 앱과 무관.
- 시간 비교는 모두 서버 KST 기준 ISO 문자열이다. 앱에서 로컬 시간으로 바꾸지 말고 그대로 보여준다.

## 5. 테스트 (vitest)

- `r016` 응답 파서 단위 테스트: 위 1.1~1.6의 예시 JSON 각각 + rc 2/3 케이스 + stdout에 JSON 앞뒤로 잡음이 섞인 경우(첫 `{`부터 마지막 `}`까지 잘라 파싱).
- 셸 인용 함수 테스트: 한글 이름, 공백, JSON 문자열 값(`"gate"`), 세미콜론/백틱 거부.
- 버튼 활성 규칙 테스트(실행 중/정지/외부 r003 존재/미적용 변경 조합).
- 편집 폼 테스트: `type`별 위젯 선택, int에 소수 입력 거부, dict JSON 파싱 실패 시 저장 차단, 변경된 키만 `set`에 포함.
- 메인 프로세스의 SSH exec 래퍼는 실제 서버 없이 모킹(명령 문자열과 stdin 내용만 검증).

## 6. 완료 기준

1. 제어 탭에서 실행 → `status.running`이 true가 되고 상태바에 반영된다. 정지 → `inactive`. 외부 r003이 있으면 실행 버튼이 비활성이다.
2. `_029 시장 추세(지수 ETF)` 노드에서 `ENABLE_MARKET_REGIME_GATE`를 끄고 저장하면 서버 `live_overrides.json`에 기록되고 노드에 "미적용" 표시, 재시작 후 사라진다.
3. `MARKET_REGIME_MA_FAST`에 `"a"`를 넣으면 저장 전 앱이 막고, `true`를 넣으면 서버 `rejected` 메시지가 행에 표시된다.
4. 감시목록에서 종목 하나를 지우고 저장하면 서버 파일이 바뀌고 백업 경로가 토스트로 보인다.
5. 모든 쓰기 동작이 제어 탭 이력 표에 작성자와 함께 나타난다.
