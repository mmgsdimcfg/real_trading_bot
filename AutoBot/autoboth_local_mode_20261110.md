# 노트북 메인 + 라즈베리파이 예비 운영 (2026-11-10)

이 문서는 `autoboth_architecture_prompt_20261107.md` 1장의 "r003은 윈도우에서 실행하지 않는다" 원칙을 대체한다.
라즈베리파이의 처리 지연(r016 호출마다 Python 기동, SFTP 폴링)을 없애기 위해, 평소에는 성능이 좋은 윈도우 노트북에서
r003을 돌리고 AutoBotH가 로컬 파일을 직접 읽는다. 노트북을 못 쓰는 날에는 라즈베리파이가 예전처럼 매매한다.

## 1. 구조

```
 ┌──── 윈도우 노트북 (메인, primary) ──────────────────────┐      ┌──── 라즈베리파이 (예비, standby) ───────┐
 │ AutoBotH (로컬 모드)                                    │ SSH  │ systemd --user autobot-r003.service     │
 │  - 로그/주가 파일: <root>/runtime 직접 읽기 (2초)        │◀────▶│  (standby 동안 r003 시작 거부)          │
 │  - r016_control.py: 로컬 자식 프로세스 (venv Python)    │      │ cron: g022 (standby면 g011 안 띄움),    │
 │  - 스케줄러: 틱 수집기 / 장 마감 작업 / 감시목록 복사   │      │       g015, g025 ...                    │
 │ r020_supervisor.py (분리 실행) -> r003 -> g011          │      │ r016_control.py (앱이 SSH로 호출)        │
 │ r021_jobs.py (분리 실행) -> g001 -> g025 -> g002        │      └─────────────────────────────────────────┘
 └─────────────────────────────────────────────────────────┘
```

- 코드는 두 호스트 모두 같은 `real_trading_bot/AutoBot` (git). 플랫폼 차이는 `live_trading/r019_platform.py` 한 곳.
- 계좌·appkey는 하나다. **r003은 한 호스트에서만** 돈다. `runtime/config/host_role.json`
  (`{"role": "primary"|"standby", "by", "at", "note"}`, 없으면 primary)로 표시한다.
  - standby 호스트: r003이 시작 시 rc 3으로 거부 (`R003_IGNORE_ROLE=1`이면 무시), r016 `start`/`restart`/`collector start` 거부,
    g022 cron이 g011(KIS 웹소켓, appkey당 1세션)을 띄우지 않음.
  - r003은 단일 실행 잠금을 잡은 뒤 **역할 잠금**(`runtime/config/.host_role.lock`) 아래에서 역할을 읽고, r016 `role set standby`는
    같은 잠금 아래에서 "r003 실행 중 아님"을 확인한 뒤 쓴다. 두 동작은 서로 끼어들 수 없다.
- AutoBotH는 r003 실행/재시작 직전에 **다른 호스트를 확인**한다(`AutoBotH/src/main/peer.ts`):
  r003 실행 중이면 거부, primary(또는 역할 파일 없음)면 standby로 바꾼 뒤 진행, 연결 불가면 사용자가 직접 확인했다고 누를 때만 진행,
  마지막으로 이 호스트를 primary로.

## 2. 윈도우에서 달라지는 것

| 항목 | 라즈베리파이 (Linux) | 노트북 (Windows) |
|---|---|---|
| 서비스 | systemd user unit (`r003_service.sh`) | `r020_supervisor.py` (r016 start가 분리 실행) |
| 재시작 정책 | Restart=on-failure, 60초, 10분 3회 | 같음 (rc 3 = 이중 실행/standby 거부는 재시작 안 함) |
| 정지 | SIGTERM -> live_state 저장, 90초 후 SIGKILL | 정지 요청 파일 `live_runtime/r003.stop` (+`r020.stop`) -> r003이 매 턴 확인, live_state 저장 후 Ctrl+C 경로로 종료. 90초 후 실제 r003(venv 런처의 자식) 강제 종료, 잠금 해제 확인 |
| 실행 여부 판단 | /proc | 각 단일 실행 잠금 파일 (`.r003.lock`, `.g011.lock`, `.r020.lock`, `.r021.lock`) - 첫 줄 `<pid> <날짜 시각> ppid=<n>` |
| 파일 잠금 | fcntl.flock | msvcrt.locking (1 GiB 지점 1바이트 - 내용은 읽을 수 있음) |
| 타임아웃 | SIGALRM | 타이머 스레드 + interrupt_main, 스크립트 실행 시 2초 뒤 os._exit(3) |
| 분리 실행 | setsid | CREATE_NO_WINDOW + 새 프로세스 그룹 + Job 탈출, 탈출이 막히면 WMI `Win32_Process.Create` |
| cron | g022 / g015 / g025 | AutoBotH 스케줄러 (앱이 켜져 있을 때) + 데이터 작업 탭 |

- venv의 `python.exe`는 실제 인터프리터를 자식으로 띄우는 런처다. 런처를 끝내면 자식도 끝나고(런처 Job의 KILL_ON_JOB_CLOSE),
  손자(r003이 띄운 g011, r016이 띄운 r020)는 살아남는다(SILENT_BREAKAWAY_OK). 2026-11-10 노트북에서 실측.
- Python **3.12 이상** 필요 (r003이 PEP 701 f-string 사용). `pandas<3`.

## 3. r016 새 명령 (두 호스트 공통, JSON 한 덩어리)

| 명령 | 응답 | rc |
|---|---|---|
| `status` | 기존 필드 + `host: {platform, hostname, machine, backend: systemd\|process, role: {role, set, by, at, note}, runtime_root}`, `service.backend`/`service.supervisor_pid`(process), `external_r003[].ppid` | 0 |
| `role get` | `{ok, role, set, by, at, note, host}` | 0 |
| `role set primary\|standby [--note T]` | `{ok, role, ...}` / standby인데 r003 실행 중이면 `{ok:false, error}` | 0 / 3 |
| `collector start [--force]` | `{ok, started, pid, log}` / `{ok, already_running, pid}` / 창 밖·standby `{ok:false, error}` | 0 / 3 |
| `collector restart` | `{ok, signalled[], restarted, pid?, log?, window?, note?}` - g011 을 멈추고 현재 r004 로 다시 띄운다 (세션 창 밖이면 멈추기만, window=사유). 앱이 감시목록 저장 때와 5분 확인에서 구독 누락(status.collector.missing)이 있으면 부른다 | 0 / 3 |
| `collector detail [--date D] [--days N]` | [2026-11-16] 수집기 탭. `{ok, date, today, generated_at, status(=collector status, 오늘만), window{ok,reason}, role, config{TICK_COLLECTOR_AUTOSTART, TICK_COLLECTOR_FILL_FROM_SCAN, TICK_COLLECT_NXT, TICK_COLLECT_NXT_REGULAR, TICK_COLLECTOR_UNTIL, MORNING_NXT_START, AFTERNOON_NXT_END, TICK_TRIGGER_BAR_SIZES, max_subscriptions}, extra{codes,updated_at,by}, scan_date, watch[], plan{codes[], roles{watch,extra,scan}, nxt_codes, nxt_keep[], subscribed{at,krx,nxt}, starts, reconnects, errors, start, start_note}, events[{ts,kind,text}], symbols[{code,name,role(watch\|extra\|scan\|file_only),planned,nxt_keep,krx{size,mtime,first,last,rows_est}\|null,nxt{...}\|null,idle_s,stale}], days[{date,krx,nxt,size}], total_size, day_count, disk{total,free}, ticks_dir, prune_min_keep_days}` (읽기 전용) | 0 / 2 |
| `collector stop` | `{ok, stopped[], r003_running, note}` - g011 중지. r003 실행 중이면 감시자가 1분 안에 다시 띄운다 | 0 / 3 |
| `collector extra [--set CODES...]` | `{ok, codes[], updated_at, by, path}` - 감시 종목 외 추가 수집 종목(`<runtime>/config/tick_collect_extra.json`, 최대 40, 6자리). g011 은 감시 → 추가 → 스캔 채움 순으로 구독, 다음 기동부터. 형식 오류 `{ok:false, error}` | 0 / 2 |
| `collector prune --keep-days N [--apply]` | `{ok, keep_days, apply, delete[{date,size}], free_bytes, removed?[], errors?[]}` - 오래된 틱 날짜 폴더 삭제 계획 / --apply 로 삭제. 오늘 제외, N >= 30 | 0 / 2 |
| `doctor` | `{ok, problems[], host, python, executable, packages{}, code{dir,commit}, open_trading_api{path,exists,commit}, kis_config{dir,devlp_yaml,token_today}, runtime{root,exists}}` | 0 |
| `job list` | `{ok, running: {pid, started_at, name, run_id}\|null, jobs: [{name, title, steps, last}]}` | 0 |
| `job start NAME [--date D] [--workers N] [--edge-ai] [--no-collect] [--no-scan] [--no-apply] [--email] [--scheduled] [--trading-day-only]` | `{ok, name, pid, run_id, running}` / 휴장일 `{ok, skipped, reason}` / 다른 작업 실행 중 `{ok:false, error, running}` | 0 / 3 |
| `job stop NAME` | `{ok, stopped}` | 0 / 3 |
| `job log NAME [-n N]` | `{ok, name, log, lines[], state}` | 0 |
| `job data --date D` | `{ok, date, dir, exists, scannable, collected, collected_count, files_10s, files_daily, data_mtime, picks: [{code,name}]\|null, picks_file, picks_mtime, scan_log_mtime, r004: {scan_date, same_codes, applied, count, mtime}}` - [2026-11-11] 그 날짜 폴더의 수집 데이터 유무와 g002 선별 결과, r004 가 그 선별과 같은지 | 0 / 2 |
| `job dates` | `{ok, dir, dates: [{date, collected, scannable, picks}], generated_at}` (최신 날짜 먼저) / 목록 실패 `{ok:false, error}` - [2026-11-17] `<runtime>/data` 의 YYYYMMDD 폴더마다 collected(g001 `_D_collected.txt`), scannable(`*_10s.txt` 있음), picks(`_D_picks.txt`). AutoBotH 장 마감 파이프라인 달력의 저장 날짜 표시 | 0 / 3 |
| `watchlist apply --date D [--restart-collector]` | `watchlist set` 과 같은 `{ok, count, backup, path, restart_required, collector_missing, collector_note}` + `date, picks_file, for_trading_day` (+ `collector_restart` when --restart-collector and a running g011 misses new codes) / picks 파일 없음 `{ok:false, error}` - [2026-11-11] 그 날짜의 `_D_picks.txt` 를 r004 로 (헤더 scan_date / for_trading_day) | 0 / 2 |

- 작업(`live_trading/r021_jobs.py`): `collect`(g001 --date D), `history`(g025 --append D), `scan`(g002 --date D --no-export-watchlist ->
  `apply`), `apply`(r016 watchlist apply --date D: 그 날짜의 `_D_picks.txt` 를 r004 로 - 백업/이력/다음 거래일 헤더), `edge_ai`(g013 --live,
  g014, g024 --edge-ai), `universe`(g010), `postclose`(collect -> history -> hi52 -> gh52 -> hi52acct -> vwaplog -> scan -> picklog -> s1log -> apply [-> edge_ai]). [2026-11-11] 단계 선택:
  `--no-collect`(수집+이력 생략, 그 날짜에 10초봉 파일이 있어야 함), `--no-scan`(선별 생략, 선별 파일이 있어야 함), `--no-apply`(r004 반영
  생략; `--keep-watchlist` 는 옛 이름). 앞 단계 산출물이 없으면 start 가 `{ok:false, error}` 로 거부. 앱의 장 마감 파이프라인 카드는
  ① 수집 ② 선별 ③ r004 반영 체크로 이 스위치를 만든다(1+2+3 / 2+3 / 3). apply 는 선별 파일이 없거나 이 실행의 scan 보다 오래되면
  건너뛴다(step state=skipped + note, 작업은 done; 상태 JSON `note`). collect/scan/apply 실패 시 뒤 단계 생략, history / hi52 / gh52 / hi52acct 실패는
  계속. [2026-11-19] collect 실패 뒤에도 history 는 실행한다(수집된 종목의 일봉 이력 갱신 - 20261118 KIS 장애 때 차트 일봉이 전 종목
  하루 비었음). 생략된 뒤 단계는 상태 JSON 의 steps 에 state=skipped + note="<단계> failed" 로 남는다(작업 state 는 failed 그대로). [2026-11-18] `hi52` 단계(g042 --date D --evaluate, postclose 에만, collect 스위치를 따름): HI52 스윙 신호 전진 기록 - §11. [2026-11-11] `--email`(postclose/scan): scan 단계를 `g002 --notify-email` 로 실행해 선별 종목(본문)과 리포트 파일(첨부)을
  메일로 보낸다(g016, 실행 호스트의 `~/.mail/mail_config.json`; scan 단계가 없으면 보내지 않음, 전송 실패는 작업 실패가 아님 -
  로그의 `[EMAIL]` 줄). [2026-11-22] 메일을 보내는 실행은 카카오톡("나에게 보내기")으로 종목코드와 종목명만 같이 보낸다 -
  실행 호스트에 `~/.kakao/kakao_config.json` + `kakao_token.json` 이 있을 때만(AutoBotHtsH 카카오톡 탭의 "봇용 로그인 파일
  만들기", 파이는 그 두 파일을 복사), 없으면 로그에 `[KAKAO] 건너뜀` 한 줄. 인자와 상태 JSON 은 그대로다. 앱은 파이프라인 카드의 "선별 결과 메일" 체크와 설정 `postcloseEmail`(자동 실행)로 붙인다.
  `--scheduled`: 앱 스케줄러가 시작한 실행 표시(상태 JSON `args.scheduled`, 실행 내용은 같음 - 앱이 "자동 실행 / 수동 실행" 표시).
  한 번에 한 작업. 상태 `runtime/data/jobs/<NAME>.json`(args 에 collect/scan/apply/email/scheduled), 로그 `runtime/log/jobs/<YYYYMMDD>/<NAME>_<HHMMSS>.log`.
- process 백엔드 `stop`은 감시자가 없더라도 r003 잠금이 잡혀 있으면 정지 요청을 보내고, 잠금이 풀릴 때만 성공한다.
- 예전 서버(이 명령들이 없는 r016)에는 `invalid choice` 오류가 난다. 앱은 이를 "git pull 필요"로 안내한다.

## 4. 운영 절차

### 평소 (노트북 메인)
1. 앱 제어 탭 맨 위 "매매 실행 위치" 표에서 이 PC 줄이 "실행 대상"인지 확인 (아니면 [여기서 실행] - runMode 저장 후 다시 연결. 설정/연결 탭의 실행 위치와 같은 값).
2. 제어 탭 [이 PC에서 실행] -> 라즈베리파이가 standby로 바뀌고 노트북이 primary가 된 뒤 r003 시작.
3. 아침 틱 수집기, 장 마감 작업(기본 20:10, r003 종료 후), 감시목록 라즈베리파이 복사는 스케줄러가 한다 (앱이 켜져 있을 때).
4. 장중 감시 종목 추가/삭제: 종목 보드 > 감시목록 편집 > 저장 -> "재시작" 확인 -> 새 목록으로 감시 재시작.

### 노트북을 못 쓰는 날 (라즈베리파이 예비)
1. (가능하면) 노트북 앱에서 r003 정지.
2. 라즈베리파이를 primary로: 앱 제어 탭 "매매 실행 위치" > 고급 > 라즈베리파이 [primary 로] (3의 [실행]이 자동으로 해 주므로 보통 생략), 또는 SSH에서
   `cd /data/srv/autobot/real_trading_bot/AutoBot && ~/venv/bin/python live_trading/r016_control.py role set primary`.
3. "매매 실행 위치" 표에서 라즈베리파이 줄의 [여기서 실행] -> [라즈베리파이에서 실행] (노트북이 켜져 있으면 노트북을 standby로 바꾼다).
4. 노트북이 꺼져 있었다면, 다시 쓰기 전에 노트북 역할을 확인한다 (앱이 [실행] 때 자동으로 확인).

### 과거 날짜 데이터 보충 / Edge AI 재학습 (장중에도 가능)
1. 데이터 작업 탭 > 장 마감 파이프라인, 기준일 = 과거 날짜. "감시목록(r004) 갱신"은 과거 날짜면 자동으로 꺼진다 - 수집 -> 이력 -> 선별까지
   돌지만 선별은 그 날짜 폴더의 `_D_picks.txt`/리포트만 만들고 지금 감시 중인 종목(r004)은 바뀌지 않는다 (g013 이 전날 picks 를 읽으므로 선별은 필요).
2. 날짜가 여럿이면 날짜마다 1을 반복한다 (한 번에 한 작업). Edge AI 는 마지막에 한 번만: "Edge AI 재학습" 작업 (g013 은 아직 안 만든 날만 추가).
   또는 마지막 날짜의 파이프라인에 "Edge AI 포함".
3. 오늘 날짜로 돌리면 갱신이 켜진 채 시작되고, r003 실행 중이면 감시목록을 덮어쓴다는 확인을 먼저 묻는다.

### 데이터
- 감시목록(r004)은 장 마감 작업 후 라즈베리파이로 복사된다. 다음 날 라즈베리파이로 매매해도 같은 종목을 감시한다.
- 일봉/분봉 데이터(`runtime/data/YYYYMMDD`)와 Edge AI 모델은 호스트별로 따로 쌓인다. 라즈베리파이가 오래 예비로 있으면
  r012(전일 분봉 예열)는 노트북이 보낸 감시목록 종목의 전일 파일이 없을 수 있다 - 필요하면 라즈베리파이에서도 장 마감 작업을 돌린다
  (원격 모드의 데이터 작업 탭).
- 보유 종목은 계좌 잔고(sync_positions_from_account)에서 복원된다. [2026-11-11] 감시목록(r004)에 없는 보유 종목은 봇이 사지도
  팔지도 않는다(HOLD SKIP NOT_IN_WATCHLIST, r001 MANAGE_POSITIONS_OUTSIDE_WATCHLIST) - 감시목록에서 뺀 종목의 청산은 사용자가 한다.
  r003은 당일 매수분만 매도 평가하므로(HOLD SKIP)
  날짜가 바뀐 뒤의 호스트 전환은 live_state를 옮기지 않아도 된다. **장중 전환은 피한다** (당일 live_state가 다른 호스트에 있음).

## 5. 노트북 준비 (한 번)
1. Python 3.12: `winget install Python.Python.3.12 --scope user`
2. venv: `<root>\venv` 생성 후 `pip install "pandas<3" numpy requests pyyaml websockets pycryptodome holidays lightgbm numba`
   (`AutoBot/requirements.txt`).
3. `git clone https://github.com/koreainvestment/open-trading-api <root>\open-trading-api` - 라즈베리파이와 같은 커밋으로
   (`r016 doctor`의 `open_trading_api.commit` 비교, 앱의 "두 호스트 환경 비교").
4. KIS 설정: 앱 [KIS 설정 가져오기] -> `%USERPROFILE%\KIS\config\kis_devlp.yaml`.
5. 전원: 덮개 닫힘 = 아무 것도 안 함, Windows Update 사용 시간에 장 시간 포함. 앱은 r003 실행 중 절전만 막는다.

## 6. r016 `bars` - NXT 시간대 분봉 (2026-11-12)
- 배경: 틱 수집기(g011)와 KIS 분봉 폴백(r018)은 KRX(J) 기준이라 09:00 이전(NXT 프리마켓)과 15:30 이후에는 분봉이 없었다.
  AutoBotH 타임라인/차트 탭이 그 시간에 비었다.
- `bars --kind min --source auto`(기본): 틱/10초 파일이 없거나 봉이 0개이면 KIS 분봉(J) -> 없으면 **KRX+NXT 통합 분봉(UN)**.
- `bars --kind min --source kis-un`(신규): 통합 분봉(08:00~20:00)을 직접 요청. 앱은 NXT 종목의 분봉 차트에서 KRX 장외 시간
  (09:00 이전, 15:30 이후)의 봉만 골라 틱 봉 아래에 합친다. `--source kis` 는 종전대로 J(틱 수집 공백 메우기).
- 응답 형식은 같다. 통합 분봉이면 `source` 가 `kis:<tr>:UN` 이다. 거래량 0 인 분(체결 없음)은 빠진다 - NXT 비대상 종목은
  장전에 봉이 없다(종전과 같은 오류 응답).
- 캐시: `runtime/cache/kis/min_un/<date>/<code>.json` (J 캐시 `min/` 과 분리).
- 틱봉(`--kind tick`)은 그대로 틱 파일이 필요하다. 앱은 틱 파일이 없으면 1분봉으로 대신 그린다(09:00 이전 포함).
- 봇 판단(r003/r013)과 틱 수집기 기준은 바뀌지 않는다. 표시 전용 경로다.
- 이전 서버(이 변경 전 r017)에 `--source kis-un` 을 보내면 argparse 오류가 난다 - 앱은 그 경우 통합 분봉 없이 그린다.

## 7. 매매 방식 선택 / 가상 매매 (2026-11-12)
- 앱 제어 탭의 "매매 방식"은 r016 `set` 으로 두 값을 함께 저장한다(r003 재시작 후 적용).

  | 선택 | TRADING_STRATEGY_MODE | EDGE_ONLY_MODE |
  |---|---|---|
  | 현재 방식 (규칙 기반) | normal | off |
  | 현재 방식 + Edge AI 가상 매매 | normal | paper |
  | Edge AI 방식 (실제 주문) | edge | off |

- 상태는 `get TRADING_STRATEGY_MODE EDGE_ONLY_MODE` 의 effective / running / pending 으로 읽는다. EDGE_ONLY_MODE 를 모르는
  서버(이 변경 전)는 `unknown` 에 넣어 돌려주므로 앱은 가상 매매 / Edge AI 방식 선택을 막는다.
- `TRADING_STRATEGY_MODE=edge`: r005 `EDGE_BUY_CONDITIONS`(_001~_008 + `_e09_edge_ai_signal` + _023~_025), r006
  `EDGE_SELL_CONDITIONS`(`_e10` 익절 +2.0% / `_e11` 손절 -1.7% / `_e12` 60분). r015 `conditions` 가 이 목록을 내보내므로
  파이프라인 탭도 edge 조건으로 바뀐다. 로그 사유: `EDGE_SIGNAL_*`(매수 반려), `EDGE_TP_*` / `EDGE_SL_*` / `EDGE_TIME_*`(매도).
- `EDGE_ONLY_MODE=paper`: r022 가 `runtime/data/live_runtime/edge_only_<YYYYMMDD>.jsonl` 에 한 줄씩 쓴다. 앱 가상 매매 탭이
  파일을 직접 읽는다(로컬 모드는 그 PC 파일, 원격 모드는 SFTP).
  - `{"event":"ENTRY","ts","code","name","entry","price","prob","threshold","trigger","tp_px","sl_px","horizon","model_id"}`
  - `{"event":"EXIT","ts","code","name","reason"(tp|sl|time|eod),"entry","exit","net_pct","held_min","prob"}`
  - 로그 태그 `[EDGE ONLY   ]` (PAPER BUY / PAPER SELL). edge 방식일 때는 가상 매매를 돌리지 않는다.
  - [2026-11-19] 틱 수집기 종목: r001 `EDGE_ONLY_COLLECTOR`(기본 True)면 감시목록/보유 밖의 수집 종목(g011 선별 채움)도 같은 규칙으로
    가상 매매한다. 1분봉·현재가는 그날 KRX 틱 파일(`runtime/data/ticks/<날짜>/<종목>.csv`)로 만든다(KIS 호출 없음, 5초마다).
    상한은 따로(`EDGE_ONLY_COLLECTOR_MAX_POSITIONS` 5 / `_MAX_ENTRIES_PER_DAY` 20). ENTRY/EXIT 에 `"group": "watch" | "collector"`
    (없으면 watch). 로그 줄에 ` [수집기]`. 앱 가상 매매 탭은 전체/감시 종목/수집기 종목으로 나눠 본다(2.3.31).
  - [2026-11-19] 수정: 2026-11-12(EDGE_ONLY_MIN_PROB=0) 이후 모델 기준값을 모델 로드 전에 읽어 가상 매수가 한 건도 없었다.
  - [2026-11-19] 상태 파일 `runtime/data/live_runtime/edge_only_status_<YYYYMMDD>.json` (r022 write_status, r003 이 정규장에 5초마다
    원자적 교체): `{"asof","date","model_id","threshold","label":{tp,sl,horizon},"entry_window":[s,e],"force_exit",
    "groups":{"watch"|"collector":{"open","max_positions","entries","max_entries","enabled"}},
    "symbols":[{"code","name","group","price","bar"(HH:MM 마지막 확정 1분봉),"bars","stage","prob","at",
    "position"?:{"entry","tp","sl","ts","prob"}}]}`. stage = wait_bars / outside_window / limit / held / below_mid / no_trigger /
    no_history / no_model / below_threshold / bought / sold / fund / sl_today / candidate (마지막 판단 결과; candidate = log-only 후보 기록; [2026-11-20] fund = ETF·ETN 제외, sl_today = 당일 손절 종목 재매수 안 함). 10분 넘게 안 본 종목(보유 제외)은 빠진다.
  [2026-11-20] 상태 파일에 `log_only` (r001 `EDGE_ONLY_LOG_ONLY`). 켜져 있으면 신규 가상 매수 없이 트리거 봉마다
  `live_runtime/edge_only_candidates_<YYYYMMDD>.jsonl` 에 CANDIDATE 한 줄(schema_version 1: id, code, group, bar, features 55개, prob,
  threshold, model_id, would_buy, block fund|sl_today, reason, price, tick, tick_pct, label) - 앱은 읽지 않는다(연구용).
    앱 가상 매매 탭 "가상 매매 대상" 표(2.3.32)가 읽는다.
- 두 호스트 공통: 설정은 각 호스트의 `runtime/config/live_overrides.json` 에 따로 저장되고, 모델도 각 호스트의
  `runtime/edge_ai` 것을 쓴다.

## 8. 자동 매도 범위 / 골든크로스 변동성 게이트 (2026-11-12)
- 앱 제어 탭의 "자동 매도"는 r016 `set AUTO_SELL_MODE=<값>` 으로 저장한다(r003 재시작 후 적용). 상태는
  `get TRADING_STRATEGY_MODE EDGE_ONLY_MODE AUTO_SELL_MODE` 의 effective / running / pending 으로 읽는다. 키를 모르는 서버는
  `unknown` 에 넣어 돌려주므로 앱은 선택을 막는다.

  | 선택 | AUTO_SELL_MODE | 봇이 하는 매도 |
  |---|---|---|
  | 전체 자동 | all | TRADING_STRATEGY_MODE 의 매도 조건 전부 + 예약 청산(15:20 / NXT 19:59) |
  | 손절만 자동 (기본값) | stop_only | 수수료+거래세 포함 순손익 <= -AUTO_SELL_STOP_NET_PCT(2%) 일 때 잔량 전량 시장가 매도 하나 |
  | 끔 | off | 없음 (손절 포함 모든 매도를 사용자가 직접) |

- `stop_only` / `off` 는 모든 TRADING_STRATEGY_MODE 에 우선하고 예약 청산도 하지 않는다(포지션이 다음 날로 넘어간다).
  사용자가 직접 판 수량은 계좌 동기화에서 빠진다.
- r015 `conditions` 출력에 `auto_sell_mode` 필드가 추가되고, `sell` 목록이 모드에 따라 바뀐다:
  `stop_only` = `_001`, `_002`, `_s04_net_stop_loss`(no 4) / `off` = `_001`, `_002`.
- 새 로그: `[SELL TRIGGER] ... | NET_STOP_LOSS_2.0PCT | price= entry= pnl= net=` (앱 매도 `_004`),
  `[AUTO SELL   ] scheduled liquidation skipped | AUTO_SELL_MODE=...` (하루 1회 안내, 조건 아님).
- 매수 `_012`(최소 변동성)는 1분봉 ATR% 를 먼저 본다(`ENABLE_GC_VOLATILITY_GATE`, `GC_VOLATILITY_MIN_ATR_1MIN_PCT`=0.5).
  반려 사유 `GC_VOL_LOW_1MIN_ATR_<값>%_LT_<기준>%` / `GC_VOL_1MIN_ATR_UNAVAILABLE` (앱 매수 `_012`), r015 frame `1m+3m`.

## 8-1. 봇 매수분 / 직접 매수분 구분 (2026-11-13)
- r003 은 계좌 동기화마다 r023 장부(`<runtime>/data/live_runtime/bot_holdings.json`, 날짜를 넘겨 유지)를 갱신한다:
  종목별 `acct_qty`(계좌 수량) / `bot_qty`(봇이 매수한 수량). 직접(MTS/HTS) 매수 수량 = acct_qty - bot_qty.
  - 계좌 수량 증가: 그 종목에 봇 매수 주문이 걸려 있던 구간(주문이 사라진 뒤 동기화 2회까지)이면 봇, 아니면 직접.
  - 계좌 수량 감소: 봇 매도 주문 구간이면 봇 수량부터, 아니면(직접 매도) 직접 수량부터 줄인다. bot_qty 는 0..acct_qty.
  - 장부 파일이 없는 첫 실행: 오늘 또는 최근 10일 live state 에 봇 매수 흔적(traded_today / buy_session != synced / reentry.bot)이
    있는 보유 종목은 계좌 수량 전부를 봇, 나머지는 직접으로 seed (`[BOT LEDGER  ] seeded ...` 로그).
  - 한계: 봇 매수 주문이 걸려 있는 동안 같은 종목을 직접 사면 둘 다 봇 수량으로 잡힌다(계좌는 합계만 알려 준다).
  - [2026-11-25] 위 세 규칙(계좌 수량 증감으로 추정)은 체결 내역 기준으로 바뀌었다. `bot_holdings.json` 의 모양(acct_qty / bot_qty)은
    그대로이고 앱 변경은 없다. bot_qty 는 r003 의 `bot_fill_ledger.json`(주문일+주문번호별 체결, KIS 주식일별주문체결조회)에서 나온다:
    봇 주문의 체결만 봇 수량, 직접 주문의 체결은 직접 수량. 직접 매도는 직접 수량부터 줄인다. 체결 순서를 알 수 없으면 봇 수량이
    작아지는 쪽으로 잡는다. 첫 실행은 기존 `bot_holdings.json` 의 bot_qty 를 이어받고 `bot_holdings.pre_fill_<날짜>.json` 으로 백업한다.
    bot_only 매도는 주문 직전에 계좌와 체결을 다시 읽고, 체결 조회가 실패하면 그 매도를 하지 않는다.
    상세·잔여 위험: `r023_fill_ledger_plan_20261125.md`.
- r001 `AUTO_SELL_SCOPE` (r016 `set AUTO_SELL_SCOPE=<값>`, r003 재시작 후 적용. 앱 제어 탭 "자동 매도 대상"):

  | 선택 | AUTO_SELL_SCOPE | 자동 매도 수량 |
  |---|---|---|
  | 봇 매수분만 (기본값) | bot_only | 봇이 매수한 수량까지만. 직접 매수분은 남긴다. 봇 수량 0 인 보유분은 매도 평가 없음 |
  | 직접 매수분 포함 | all | 종전 동작 - 봇이 관리하는 종목은 계좌 수량 전부 |

  관리 범위(오늘 감시 목록에 있고 오늘 매수된 종목만)는 어느 쪽이든 그대로다. dry_run 에는 적용하지 않는다.
  앱은 `get ... AUTO_SELL_SCOPE` 의 effective / running / pending 으로 읽는다(키를 모르는 서버 = 선택 숨김).
- 새 r016 명령 `holdings` (읽기 전용, KIS 호출 없음 - r003 이 계좌 동기화마다 쓰는 `live_runtime/holdings.json` 을 돌려준다):
  ```json
  {"ok": true, "available": true, "running": true, "age_seconds": 12.3, "schema": 1,
   "updated_at": "2026-11-13T10:00:00", "date": "20261113", "auto_sell_scope": "bot_only",
   "holdings": [{"code": "222800", "name": "심텍", "qty": 8, "bot_qty": 3, "manual_qty": 5, "avg_price": 130000.0,
                 "current_price": 131000.0, "buy_time": "2026-11-13T09:48:30", "bought_today": true}]}
  ```
  스냅샷 파일이 없으면 `{"ok": true, "available": false, "running": <bool>, "holdings": []}`. `avg_price` 는 계좌 평균단가
  (봇+직접 합산), `current_price` 는 동기화 시점 가격. r003 이 꺼져 있으면 마지막 스냅샷과 `age_seconds` 를 그대로 준다.
- 새 로그: `[🔴HOLD SKIP ] ... | NOT_BOT_POSITION (...)` (bot_only 에서 봇 수량 0 인 보유분 - 앱 종목 보드 '보유'),
  `[SELL LIMIT  ] ... | AUTO_SELL_SCOPE=bot_only | requested= bot= account=` (매도 수량을 봇 수량으로 줄임),
  `[BOT LEDGER  ] ... | account=a->b bot=c->d` (장부 변화). 뒤 두 태그는 조건이 아니다(앱에서 무시).

## 9. 틱 수집기 NXT 체결 (2026-11-12)
- g011 이 같은 웹소켓 세션에서 NXT 체결(H0NXCNT0)도 받는다(r001 `TICK_COLLECT_NXT`, 기본 True). 감시목록 종목을
  08:00~20:00 구독하되 세션 한도 40 에서 KRX 구독이 쓰고 남은 자리만큼만 받는다(정규장 중 여유분, 장전·장후 전부).
  `--fill-from-scan` 은 감시목록 종목 수만큼 NXT 자리를 남기고 채운다.
- 파일: `runtime/data/ticks/<YYYYMMDD>/nxt/<code>.csv` - KRX 파일(`<code>.csv`)과 같은 열, `venue=NXT`,
  `session=pre|regular|post`. 수집 전용이다: r013 틱 트리거, r017 틱봉, 앱 틱 차트는 KRX 파일만 읽는다.
- r016 `collector start` 의 허용 시간: 기존 (장 시작 15분 전 ~ 마감) 에 NXT (07:55 ~ 20:00) 를 더한 범위. 창 밖 거절 응답의
  `window` 문구에 NXT 구간이 붙는다. `collector status` 응답 형식은 그대로다(nxt 폴더는 집계하지 않음).
- 앱 스케줄러의 수집기 확인 시간대: 평일 07:55~19:55 (기존 08:40~15:35).
- g011 로그 추가 줄: `NXT codes=N window=08:00~20:00 ...`, `SUBSCRIBED krx=A nxt=B (+x -y)`, `NXT SUBSCRIBE ERROR <code> ...`,
  `STOPPED rows=... nxt_rows=...`. `START` / `CODES` 줄은 그대로(KRX 구독 목록).

### 9-0. 운용 시간 설정 (2026-11-20)
- r001 `ENABLE_NXT_SESSION` 하나가 "운용 시간"이다 (앱 설정 탭: 정규장만 / 프리·애프터 포함, r016 `set` / `unset`). 응답 형식 변경 없음.
- 끄면(정규장만): r003 15:30 종료(종전 그대로), g011 NXT 구독 안 함(`TICK_COLLECT_NXT` 가 True 라도), r016 `collector_nxt_hours()` = None
  -> `collector start` 허용 시간이 KRX (장 시작 15분 전 ~ 마감) 만. r022 가상 매매는 설정과 무관하게 정규장만.
- 앱: 스케줄러 수집기 확인 08:45~15:35, 장 마감 작업 `settings.postcloseTimeRegular` (기본 15:40) / 켜면 07:55~19:55, `postcloseTime` (20:10).
  차트·타임라인 x축 09:00~15:30 (켜면 08:00~20:00). 판단 기준: 실행 중 r003 값, 없으면 저장값, 모르면 켬.
- 라즈베리파이 cron 은 저장소 밖이다: g022 는 g011 이 직접 r001 을 보므로 따라가지만, 고정 시각 cron 이 있으면 손으로 맞춘다.

### 9-1. 틱 CSV v3 - 호가/수급 열 (2026-11-17)
- KRX/NXT 틱 CSV 끝에 9열 추가(r009 `TICK_DEPTH_COLUMNS`, 기존 열 순서 불변): `askp_rsqn1,bidp_rsqn1,total_askp_rsqn,
  total_bidp_rsqn,cttr,seln_cntg_smtn,shnu_cntg_smtn,wghn_avrg_stck_prc,vi_stnd_prc` (1호가 잔량, 총 매도/매수 호가 잔량,
  체결강도, 누적 매도/매수 체결량, 가중평균가, VI 기준가). 체결 레코드에 이미 있는 값이라 구독 수는 그대로다.
- 새 파일부터 v3. 이미 있는 파일은 그 헤더 폭대로 계속 쓴다(같은 날 장중 재시작에도 한 파일에 폭이 섞이지 않음).
  앱(`price.ts`)과 r009/r013/r017 판독기는 헤더 이름으로 열을 찾으므로 변경 없음.

## 10. 일봉 시장 추세 게이트 / 과거 일봉 백필 (2026-11-17)
- r005 `_010` 앞단 일봉 게이트(r001 `ENABLE_MARKET_DAILY_TREND_GATE`, 기본 꺼짐): 지수 ETF(069500, 229200) daily_history 일봉이
  어제 종가 > MA60 이고 MA60 이 10거래일 전보다 높을 때만 신규 매수. 반려 로그
  `[REJECT] <종목> | MARKET_REGIME_DOWN_DAILY_MA60 | 069500_CLOSE_BELOW close=.. ma60=.. ma60_10d_ago=.. asof=..` -
  기존 접두어 `MARKET_REGIME_DOWN` 그대로라 앱 매핑은 `_010`. 설정 키는 `r016 conditions` 의 `_010` defines 에 포함.
- g040(과거 일봉 백필): `<runtime>/data/daily_backfill/<code>.csv` (+ `_progress.json`) - daily_history 와 별도 폴더, 앱은 읽지 않는다.
  파이에서 장 외 시간(평일 20:10~07:40, 주말 종일)에 2회/초로 받고, 노트북은 `g041_pull_daily_backfill.ps1 -PiHost ..` 로 가져온다.

## 11. HI52 스윙 신호 기록 (2026-11-18)
- r021 `postclose` 의 새 단계 `hi52` = `data_simulation/g042_hi52_signal_log.py --date D --evaluate` (history 다음, scan 앞, 실패해도
  작업은 계속, `--no-collect` 면 생략). 주문 없음, r003/r004 불변. 작업 목록 JSON 의 steps 에 `"hi52"` 가 추가된다 - 앱 단계 라벨 "HI52 기록".
- (2026-11-19) postclose 작업 title = "장 마감 파이프라인 (데이터 수집(g001) -> 종목 선별(g002) -> 감시목록 반영(r004))" - 앱 단계 체크박스 이름과
  같다(스크립트 코드는 이름 뒤 괄호). ① 데이터 수집(g001) = collect + history(g025) + hi52(g042). steps 이름은 그대로(표시 문구만 변경).
  종료 코드: 0 정상(주말 포함) / 2 입력 없음(일봉 파일·지수 ETF 파일 없음) / 3 다른 g042 실행 중(`.g042.lock`) / 4 평일인데 D 의 지수 ETF
  일봉이 없음(휴장일 또는 g025 미반영) - 단계 rc 로 앱에 그대로 보인다.
- 산출물 `<runtime>/data/hi52_signals/`: `days.csv` = 기록 원본(날짜별 지수 게이트 UP/DOWN/NA, 신호 수, provisional, recorded, 그날
  신호 목록 JSON - 파일 하나의 원자적 교체가 커밋), `signals.csv`(days.csv 에서 다시 만드는 보기용 평면 표), `evaluation.csv`(다음날
  시가 매수 -> 20/40거래일 종가 매도 순수익률과 상태, 같은 날 유니버스 평균 대비 초과 - 매 실행 전체 재계산), `report.md`.
  앱은 아직 읽지 않는다. `g042 --status` 는 한 줄 JSON(last_date, last_live, last_gate, signals, report_at).
- recorded: `live` = 데이터의 최신 거래일을 다음 평일 09:00 전에 기록(전진 검증은 이것만), `late` = 지난 날짜 재실행 또는 개장 후 기록,
  `backfill` = `--backfill-from`. live 날짜는 late/backfill 실행이 덮어쓰지 않는다.
- 규칙(연구 runtime/research/08_daily_swing 와 같음): 종가 >= 2,000, 20일 평균 거래대금 >= 50억, 최근 60행에 절댓값 30.5% 초과 이동이나
  시가 갭 없음, 지수 ETF 제외, 종가가 직전 250(500)행 최고 종가 이상이고 상승일; 게이트 = 069500/229200 모두 종가 > MA60 이고 MA60 이
  10행 전보다 높음. 평가: 보유 구간에 30.5% 초과 이동이 있으면 제외, 청산일 종가가 없으면(정지/폐지) 구간 안 마지막 종가로 청산.

## 12. 선정 종목 성과 기록 (2026-11-19)
- r021 `postclose` 의 새 단계 `picklog` = `data_simulation/g047_pick_forward_log.py --date D --evaluate` (scan 다음, apply 앞, 실패해도
  작업은 계속, scan 스위치를 따름 - `--no-scan` 이면 생략). 주문 없음, r003/r004 불변. 작업 목록/상태 JSON 의 steps 에 `"picklog"` 가
  추가된다 - 앱 단계 라벨 "선정 성과 기록(g047)".
  종료 코드: 0 정상(주말 포함) / 2 입력 없음(일봉 파일·선별 파일 없음) / 3 다른 g047 실행 중(`.g047.lock`) / 4 평일인데 D 의 지수 ETF
  일봉이 없음(휴장일 또는 g025 미반영).
- 산출물 `<runtime>/data/pick_forward/`: `days.csv` = 기록 원본(날짜별 시장 필터 UP/DOWN/NA, 선정 종목 JSON - 종가, 1틱 비율),
  `evaluation.csv` = 종목별 1/5/10/20일 순익·시장 대비 초과, `report.md` = 기록 종류 x 필터 x 1틱 구간 요약. 앱은 읽지 않는다.
  배경: `selection_filter_redesign_20261119.md` 10~11장.


## 13. 장전/장후 거래 시간과 KRX 장후 분봉 (2026-11-19)
- 배경: 403870 HPSP 는 NXT 비대상(KIS `cptt_trad_tr_psbl_yn`=N, NXT 체결 0)인데 KRX 에서 장후(평일 16:00~20:00, 수능일 17:00~20:00)에
  1분 단위로 연속 체결된다. 차트가 KRX 정규장 마감에서 멈췄다. KRX 는 장전 거래가 없다(08:00~09:00 분봉 없음).
- r014 `trading_hours(day)` / `trading_hours_json(day)`: 정규장 + `nxt_pre` / `nxt_after` / `krx_after` (없는 세션은 null).
  평일 = 09:00~15:30 / 08:00~08:50 / 15:40~20:00 / 16:00~20:00. 특수일은 마감에서 파생(NXT 장후 = 마감+10분, KRX 장후 = 마감+30분,
  개장이 09:00 이 아니면 NXT 장전 없음), 달력 줄 뒤 `nxt_pre=HH:MM-HH:MM|none` 등으로 덮어쓴다. 20261119(수능일)는 KIS 분봉으로
  확인한 값(NXT 장전 없음, NXT 장후 16:40, KRX 장후 17:00)을 적었다. g022 awk 는 `$2 $3` 만 읽어 영향 없음.
- r018 `nxt_flag(code)`: `cptt_trad_tr_psbl_yn`=Y 이고 `nxt_tr_stop_yn`!=Y 면 true. names.json 에 `nxt`/`nxt_at` 으로 7일 캐시.
  `minutes(after=True)`: KRX(J) 분봉을 KRX 장후 끝까지(캐시 `min_ax/`), 마감 뒤 거래량 0 행은 버린다. 기본 minutes() 는 정규장 그대로.
- r017 `bars --kind min --source kis-ext`(신규): **정규장 밖 분봉만** - NXT 종목 = 통합(UN, NXT + KRX 장후), 그 외 = KRX 장후(J after).
  비어 있어도 ok 응답(count 0). `--source auto` 에서 파일이 없으면 하루 전체를 같은 기준으로(NXT 비대상 = J 를 장후까지,
  NXT 종목 = 정규장 J + 정규장 밖 UN). 
- tick/min 응답 필드 추가: `hours`(위 trading_hours_json), `nxt`(true/false/null - kis-ext/auto 는 KIS 조회, 그 외는 캐시 값).
- 앱(AutoBotH 2.3.30): 분봉 차트는 **모든 종목**에 kis-ext 를 합친다(정규장 밖에서는 KIS 봉이 이긴다, 정규장 구분은 응답 `hours`).
  서버가 KRX 전용이라고 알려준 종목은 개장 전에 묻지 않는다. 지난 날짜 저장 파일은 `kis_min/<code>_ext.json`,
  `server_min/<code>_<n>[_w].v2.json`(예전 파일은 장후가 없어 새 이름). 이전 서버(kis-ext 없음)면 NXT 표시 종목만 종전 kis-un.
  타임라인의 KRX 전용 종목 표시 범위 09:00~15:30 -> 09:00~20:00. 감시목록 밖 보유 종목도 응답 `nxt` 로 NXT 표시.
- 바뀌지 않는 것: r003 매매 시각(r001 고정값 - 수능일 미반영, KRX 장후 매매 없음), 틱 수집기(g011, KRX 체결은 정규장 마감+1분까지).
  둘 다 별도 승인 사항.

## 14. 52주 고점 근접(GH52) 전진 기록 (2026-11-21)
- r021 `postclose` 의 새 단계 `gh52` = `data_simulation/g048_gh52_forward_log.py --date D --evaluate` (hi52 다음, scan 앞, 실패해도
  작업은 계속, collect 스위치를 따름 - `--no-collect` 면 생략). 주문 없음, r003/r004 불변. 작업 목록/상태 JSON 의 steps 에 `"gh52"` 가
  추가된다 - 앱 단계 라벨 "52주 고점 기록(g048)". ① 데이터 수집(g001) = collect + history(g025) + hi52(g042) + gh52(g048) + hi52acct(g049).
- D 가 그달 마지막 거래일(다음 KRX 거래일이 다른 달 - r014 달력)일 때만 코호트를 기록하고, 다른 날은 "not the last trading day" 한 줄 후
  평가만 다시 한다(rc 0). 종료 코드: 0 정상(주말·월말 아님 포함) / 2 입력 없음 / 3 다른 g048 실행 중(`.g048.lock`) / 4 평일인데 D 의
  지수 ETF 일봉이 없음(휴장일 또는 g025 미반영).
- 산출물 `<runtime>/data/gh52_signals/`: `formations.csv` = 기록 원본(월말 날짜, 유니버스 종목 목록, 상위 30% 선택 종목 JSON, recorded),
  `evaluation.csv` = 코호트별 다음 시가 매수 -> 6번째 월말 다음 시가 매도 순익, 같은 날 유니버스 대비 초과, KODEX200 참고 수익(매 실행
  전체 재계산), `report.md` = 사전 등록 판정 진행(2026-11-21 등록, live 코호트 36개) + 코호트 표. 앱은 읽지 않는다.
  `g048 --status` 는 한 줄 JSON(cohorts, last_date, last_live, report_at, acceptance_registered, acceptance_cohorts).
- 배경: runtime/research/13_public_strategies (공개 파라미터 검증, George-Hwang 2004).

## 15. HI52 스윙 모의계좌 (2026-11-21)
- r021 `postclose` 의 새 단계 `hi52acct` = `data_simulation/g049_hi52_paper_account.py --date D` (gh52 다음, scan 앞, 실패해도 작업은
  계속, collect 스위치를 따름). 주문 없음, KIS 호출 없음, r003/r004 불변. steps 에 `"hi52acct"` 추가 - 앱 단계 라벨 "HI52 모의계좌(g049)".
- 규칙(`RULE_VERSION hi52-t40-v1`, 고정): g042 HI250 신호 + 지수 게이트 UP, 후보는 sha256(계좌|날짜|종목) 순, 다음 시가에 슬롯당 전일 종가
  순자산의 1/10(10슬롯, 정수 주), 40번째 보유 행 종가에 만기 -> 다음 시가 매도, 손절 없음. RANDOM = 같은 날 같은 수의 유니버스 종목.
- 마지막 처리일 이후의 모든 행을 차례로 처리(전날 저녁 계획을 시가에 체결 -> 종가 순자산 -> 다음 시가 계획 동결, recorded live/late).
  같은 행을 다시 실행해도 바뀌지 않는다. 종료 코드: 0 정상 / 2 입력 없음 / 3 다른 g049 실행 중(`.g049.lock`) / 4 D 의 지수 ETF 일봉 없음.
- 산출물 `<runtime>/data/hi52_paper/`: `state.json`(계좌별 현금·보유·미체결 매도, last_processed), `plans.csv`(저녁에 동결한 다음 시가
  주문 목록), `ledger.csv`(체결/건너뜀과 사유, 시가, 체결가, 수수료, 세금), `nav.csv`(행별 현금·평가액·순자산·노출·매수액),
  `acceptance_final.json`(한 번만), `report.md`. 앱은 읽지 않는다. `g049 --status` 는 한 줄 JSON.
- 사전 등록 판정(2026-11-21, 900행 후 한 번): HI52-RANDOM 일수익 차이 평균의 단측 95% 하한 > 0(60행 블록), 매수액 0.25% 추가 후에도 > 0,
  HI52 MDD > -45%, 매수 체결률 >= 90%, late 계획 <= 10%. 파일럿 권고는 g042 와 g049 둘 다 PASS 일 때만(자동 매매 없음).
- 배경: runtime/research/14_swing_rules (매수 게이트 2 x 매도 규칙 7 비교 - 현행 규칙보다 나은 조합 없음).

## r016 quote - NXT 통합 시세 (2026-11-23)
- `quote` 응답에 `venue` 추가: `"J"`(KRX) 또는 `"UN"`(KRX+NXT 통합). NXT 거래종목은 거래일 08:00~20:00 중 KRX 정규장 밖에서 통합 시세로
  현재가·호가·체결을 묻는다(그 전에는 항상 J 라 KRX 개장 전에 전일 종가 0.00% 만 왔다). 정규장 중과 NXT 비대상 종목은 J 그대로.
- 앱: `venue` 가 UN 이면 현재가 줄에 NXT 표시, NXT 시간대의 quote 재사용 시간 5분 -> 20초. 표시 전용이며 봇 판단은 바꾸지 않는다.

## 모니터링 시간 / 자동매매 시간 분리 (2026-11-23)
- r001 `ENABLE_NXT_SESSION` = 모니터링 시간(r003 루프·시세·틱 수집·차트가 NXT 08:00~08:50, 15:30~20:00 도 도는지). 신규 r001
  `ENABLE_NXT_TRADING`(기본 True) = 자동매매 시간. 실주문이 NXT 세션에 나가는 조건은 두 값 모두 True (`r003.nxt_trading_enabled()`).
  `ENABLE_NXT_SESSION=True, ENABLE_NXT_TRADING=False` 면 NXT 시간에는 주문이 없다: 신규 매수 차단, `get_order_spec` None(매도 포함),
  NXT 가능 종목도 15:20 정규장 시장가 청산, 19:59 청산 없음. g003 도 같은 스위치를 따른다.
- NXT 당일 청산: `AFTERNOON_NXT_FORCE_EXIT` 19:59:58 -> 19:59:50, 가격은 매수1호가보다 `NXT_FORCE_EXIT_BID_TICKS`(3)틱 낮은 지정가 1회
  (NXT 는 시장가 없음, 호가 조회 실패 시 현재가 기준, 0 이면 종전 현재가 지정가). 정규장 청산(15:20 시장가)은 그대로.
- r016 계약 변경 없음: 앱은 `get ENABLE_NXT_SESSION ENABLE_NXT_TRADING` 한 번으로 두 값을 읽고(`values` 의 키별 base / override /
  effective / running / pending), `set` / `unset` 으로 바꾼다. 키를 모르는 옛 서버는 rc 2 + `unknown` 으로 답하므로 앱은
  `get ENABLE_NXT_SESSION` 으로 다시 묻고 자동매매 시간 스위치를 "서버 갱신 후 표시"로 둔다.
- 앱 설정 "운용 시간": 모니터링 시간(정규장만 / 08:00~20:00)과 자동매매 시간(정규장만 / 08:00~20:00). 자동매매 08:00~20:00 은
  모니터링이 08:00~20:00 일 때만 고를 수 있다. 장 마감 작업 시각은 모니터링 시간을 따른다(08:00~20:00 -> 기본 20:05, 종전 20:10;
  정규장만 -> 15:40). 저장돼 있던 20:10 은 앱이 한 번 20:05 로 옮긴다.
- 가상 매매(r022): 신규 r001 `EDGE_ONLY_NXT`(기본 True) 와 `ENABLE_NXT_SESSION` 이 모두 True 면 감시 종목 중 NXT 가능 종목을 NXT 장전
  (매수 08:05~08:35, 청산 08:45) / 장후(매수 15:35~19:30, 청산 19:50)에도 평가한다. `edge_only_<날짜>.jsonl` 의 ENTRY / EXIT 와
  `edge_only_candidates_<날짜>.jsonl` 의 CANDIDATE 에 `"session"`: `"regular"` | `"nxt_pre"` | `"nxt_after"` (없는 기록 = regular).
  `edge_only_status_<날짜>.json` 에 `"nxt"`(bool), `"nxt_sessions"`: `{"nxt_pre": {"entry_window": ["08:05","08:35"], "force_exit":
  "08:45"}, "nxt_after": {...}}` (꺼져 있으면 `{}`), 보유 행의 `position.session`. 수집기 종목(KRX 틱 파일)은 정규장만.
  모델은 정규장 데이터로 학습된 것이므로 NXT 세션 성과는 session 으로 나눠서 본다.
- 불변: 종목 선별(g002)은 정규장 09:00~15:30 행만 집계, 데이터 수집(g001)은 NXT 가능 종목이면 08:00~20:00 / 아니면 정규장만.

## 가상 매매 청산 기준 직접 설정 (2026-11-25)
- 신규 r001 `EDGE_ONLY_TP_PCT` / `EDGE_ONLY_SL_PCT` (%, float, 2.0 = +2.0%) / `EDGE_ONLY_HORIZON_MIN` (분, int). 기본 0 = 모델 라벨
  (`runtime/edge_ai/current_meta.json` label) 그대로. 0 보다 크면 r022 가 새 가상 매수의 목표 / 손절 / 보유 시간 상한으로 쓴다.
  이미 있는 가상 보유(재시작 복원 포함)는 매수 때 정한 값 그대로. r003 재시작 후 적용. 주문 없음.
- r016 계약 변경 없음: 앱은 `get` 에 세 키를 함께 묻고 `set` 으로 바꾼다(앱 범위 tp / sl 0~30, horizon 0~720 정수). 키를 모르는 옛 서버는
  rc 2 + `unknown` 으로 답하고 나머지 값은 그대로 오므로 앱은 입력칸만 숨긴다.
- `edge_only_status_<날짜>.json`: `label` = 실제 쓰는 값(설정이 반영된 값), 신규 `model_label` = 모델 값(옛 서버에는 없음).
  `edge_only_candidates_<날짜>.jsonl` 의 CANDIDATE `label` 은 모델 값 그대로(연구용, 학습 정의와 맞춤).

## 가상 매매 세션 마감 청산 (2026-11-25, trade_modes_plan_20261125.md 1단계)
- 가상 매매의 NXT 운용 = 자동매매 시간(nxt_trading_enabled()). r001 `EDGE_ONLY_NXT` 는 쓰지 않는다(상수만 남김).
- 청산: 정규장 종목(NXT 불가, 수집기 종목, NXT 안 하는 운용) = 15:20 종가 단일가, 가상 체결가 = 15:30 단일가 체결(틱 파일 session
  close_auction), 15:32 까지 없으면 틱 파일 마지막 정규장 체결 -> 마지막 가격 -> 매수 기록 가격 - 1틱(`estimated`). NXT 가능 종목(NXT 운용)
  = 19:59:50, NXT 틱 파일 3분 이내 체결 -> r003 시세 3분 이내 -> 마지막 가격 - 1틱(`estimated`). 장전/정규장/장후를 넘어 보유
  (예전 08:45 / 15:15 / 19:50 청산 없음).
- `edge_only_<날짜>.jsonl`: ENTRY 에 `"nxt"`(bool, 없는 예전 기록 = false). EXIT `reason` 에 `close_regular` / `close_nxt`(예전 `eod`
  그대로 읽음), `"fill": "auction" | "last"`, `"estimated": true`(신선한 가격 없음), `"nxt"`.
- `edge_only_status_<날짜>.json`: `"close_regular": "15:20"`, `"close_nxt": "19:59:50" | null`, `nxt_sessions.*.force_exit` = `"19:59:50"`,
  보유 행 `position.nxt`, `position.close`(그 보유의 청산 시각). `force_exit` 는 "15:20".
- 앱: 청산 사유 "정규장 종가 청산" / "NXT 마감 청산", estimated 면 "(추정)".

## 매매 방식 조합 (2026-11-25, trade_modes_plan_20261125.md 2단계)
- 앱 제어 탭 "매매 방식" = 체크박스 규칙 매매 / Edge AI 가상 매매 / 반등 알림(AutoBotHtsH 만, 앱 설정 - 바로 적용). 어떤 조합도 가능.
- 규칙 매매 = 신규 r001 `RULE_BUY_ENABLED`(bool, 기본 true). false = 규칙 기반 전략(normal / basic)의 신규 매수 주문 없음
  (`[RULE BUY OFF]` 로그). 보유 매도·미체결 정리·회계는 그대로. edge 전략에는 적용 안 함. 앱은 `get` 에 함께 묻고, 키를 모르는 서버면
  체크박스를 잠근다.
- Edge AI 가상 매매 = `EDGE_ONLY_MODE` paper / off (예전과 같음). 체크박스는 `TRADING_STRATEGY_MODE` 를 바꾸지 않는다. 서버가 edge / basic
  이면 체크박스 대신 "서버 설정" 표시 + "규칙 매매 방식으로 전환"(normal + paper off, 확인 후).
- 가상 매매를 꺼도(off / edge) 오늘 `edge_only_<날짜>.jsonl` 에 열린 가상 보유가 있으면 r003 이 close_only 러너로 그 보유만 마감까지
  청산한다. 상태 파일에 `"close_only"`(bool).

## Edge AI 익절/손절 한 곳에서 (2026-11-25, trade_modes_plan_20261125.md 3단계)
- 새 키 없음: 앱 제어 탭 "Edge AI 청산" 칸 = r001 `EDGE_ONLY_TP_PCT` / `EDGE_ONLY_SL_PCT` (%) / `EDGE_ONLY_HORIZON_MIN` (분) 이 유일한 입력처.
  r022 가상 매매(0 = 모델 라벨)와 r006 edge 전략 실주문 매도 _e10~_e12(0 = `EDGE_TP_PCT` / `EDGE_SL_PCT` 비율 / `EDGE_MAX_HOLD_MINUTES`)가 같이 쓴다.
  r006 은 매 평가 때 읽음 - r003 재시작 후 이미 보유한 종목에도 적용(edge + AUTO_SELL_MODE all 일 때만). 매도 사유 `EDGE_TP_1.54PCT` 처럼 설정 정밀도.
- 규칙 매매(r006 normal / basic 다단계, HARD_STOP, AUTO_SELL_STOP_NET_PCT, 적응형 엔진)와 반등 알림에는 적용하지 않는다(사용자 결정).
- 앱은 가상 매매 또는 edge 전략일 때 칸을 보인다. 서버 r006 이 예전 코드면 실주문에는 반영되지 않으므로 앱과 서버 코드를 같이 갱신할 것.

## 지난 날짜 미청산 가상 보유 정리 / 세션 보고서 (2026-11-25, trade_modes_plan_20261125.md 8절 B·C)
- r003 이 가상 매매 러너를 만들 때(가상 매매가 꺼져 있어도) r022 `close_missed_days()` 가 지난 달력 7일의 `edge_only_<날짜>.jsonl` 에 남은
  가상 보유를 그 날짜 파일에 EXIT 로 정리: `reason` `close_missed`, `ts` = 그날 예정 청산(15:20 / NXT 19:59:50), `price_at` (쓴 체결의 HHMMSS),
  `recorded_at`, `fill` `auction` | `last` | `entry_record`, 신선한 가격이 아니면 `estimated`. 앱 표시 "미청산 정리".
- `data_simulation/g050_paper_session_report.py`: 가상 매매 세션별 성과 보고서(`runtime/research/paper_session/`). 앱은 읽지 않음.

## r003 종료·재시작 시 봇 미체결 주문 (2026-11-25, trade_modes_plan_20261125.md 9절 A1+A2)
- 앱 계약 변경 없음(앱은 새 파일을 읽지 않음). r016 stop / restart(r003.stop), Ctrl+C, SIGTERM 으로 끝날 때 r003 이 봇 미체결 주문을 취소하고
  최대 40초 정리(A1). 남은 주문·응답 불명 주문은 `runtime/data/live_runtime/order_holds_<날짜>.json`
  (`{"date", "written_at", "holds_all", "orders": {code: {side buy|sell|both, order_nos, quantity, qty_by_side, ...}}}`)에 남고, 같은 날 다음
  r003 은 그 종목 주문을 보류(A2) - 브로커 미체결 목록(정정취소가능주문조회)이 정상 조회되고 열린 주문이 없을 때 다음 계좌 동기화 뒤 해제.
  파일을 못 읽으면 전 종목 보류 후 브로커 목록으로 대체. 로그 태그 `[STOP ORDERS ]` `[STOP CANCEL ]` `[ORDER HOLD  ]` `[ORDER BLOCK ]`.
- 보류 종목의 계좌 수량 증가는 봇 몫으로 잡지 않는다(직접 보유, 로그로 확인 요청), 감소는 봇 몫에서 먼저(r023). 재시작은 가능하면
  미체결이 없을 때 / 장 마감 후.

## Edge AI 실주문 병행 (2026-11-25, edge_live_parallel_plan_20261125.md)
- r001 `EDGE_LIVE_ORDERS`(bool, 기본 false): true = 규칙 매매와 함께 Edge AI 신호로 실제 주문. `TRADING_STRATEGY_MODE=edge`(단독)와는 별개.
  한도 키(정수): `EDGE_LIVE_ORDER_KRW`(1회 금액, 기본 100000) / `EDGE_LIVE_MAX_POSITIONS`(1) / `EDGE_LIVE_MAX_ENTRIES_PER_DAY`(1) /
  `EDGE_LIVE_DAILY_LOSS_KRW`(10000). 모두 r003 재시작 후 적용. 신규 매수 마감 14:50(`EDGE_LIVE_ENTRY_END`, 앱에서 바꾸지 않음).
- r016 `set` 추가 검증(거부 사유는 `rejected`에):
  - 한도 범위: ORDER_KRW 10,000~1,000,000 / MAX_POSITIONS 1~5 / MAX_ENTRIES_PER_DAY 1~20 / DAILY_LOSS_KRW 1,000~1,000,000.
  - `EDGE_LIVE_ORDERS=true` 는 `AUTO_SELL_MODE=all` 이고 `TRADING_STRATEGY_MODE=normal` 일 때만. 켜져 있는 동안 `AUTO_SELL_MODE` 를
    stop_only / off 로, `TRADING_STRATEGY_MODE` 를 다른 값으로 바꾸는 것도 거부.
  - 자동 해제: `EDGE_ONLY_MODE` / `AUTO_SELL_MODE` / `TRADING_STRATEGY_MODE` 가 `set` / `unset` / `reset` 으로 바뀌면 저장된
    `EDGE_LIVE_ORDERS=true` 를 false 로 바꾼다(응답 `edge_live_disarmed: true`, 같은 명령에 `EDGE_LIVE_ORDERS=true` 가 있으면 제외).
    실제 매매는 `set EDGE_LIVE_ORDERS=true` 로만 켜지고, 조건 키를 되돌려도 저절로 살아나지 않는다.
- r003: Edge 가 이미 산 수량의 청산(손절 / 익절 / 시간 상한 / 15:20)은 `AUTO_SELL_MODE` 와 무관하게 항상 한다(unset / reset 으로도 남지 않음).
  신규 Edge 매수는 `EDGE_LIVE_ORDERS=true` + `EDGE_ONLY_MODE=paper` + `AUTO_SELL_MODE=all` 일 때만. 앱은 "Edge AI 매매"를 끄면 `EDGE_LIVE_ORDERS=false` 도 함께 저장하고,
  고급 설정에서 `EDGE_LIVE_ORDERS` 를 켤 때도 같은 경고 확인을 거친다.
- r016 `status` 에 `edge_live` 추가(체결 장부가 없으면 null):
  `{"held": [{"code", "qty", "avg_price", "entry_at"}], "open_buys": ["종목코드"], "today": {"entries", "realized", "sl_codes", "unpriced_sells"}}`.
  realized = 그날 Edge 매도의 비용 포함 실현 손익(원). unpriced_sells > 0 이면 손익 미상으로 그날 신규 Edge 매수 중단.
- 앱 제어 탭(2026-11-25 구조): 청산 기준(맨 위, `EDGE_ONLY_TP_PCT` 등 - Edge AI 실제·가상 공통) / 매매 방식 = 규칙 매매(`RULE_BUY_ENABLED`),
  Edge AI 매매(`EDGE_ONLY_MODE`) 와 그 아래 실제 매매(`EDGE_LIVE_ORDERS`) · 가상 매매(`EDGE_ONLY_LOG_ONLY` 의 반대) / 반등 알림(별도 묶음,
  AutoBotHtsH). 실제 매매가 켜져 있으면 가상 매매는 후보만 기록. 서버에 키가 없으면(구형) 실제 매매 체크는 잠김.
- 규칙 매매 청산 기준(2026-11-25 사용자 결정, r006 `_c03_common_exit_bounds`): 새 키 `RULE_EXIT_TP_PCT`(float, %) /
  `RULE_EXIT_SL_PCT`(float, %) / `RULE_EXIT_HORIZON_MIN`(int, 분), 기본 0. 0 보다 큰 값은 규칙 매매 보유분에 상한·하한으로 적용 -
  기존 규칙 청산은 그대로, 먼저 닿으면 잔량 전부 매도(사유 `COMMON_TP_*` / `COMMON_SL_*` / `COMMON_TIME_*`). 정규장에서만,
  `AUTO_SELL_MODE=all` 일 때만. 0 인 항목은 미적용. Edge AI 쪽 `EDGE_ONLY_TP_PCT` 등과는 별개 값(Edge 는 종전대로 0 = 모델 값 / 서버 기본).
  앱 제어 탭 "청산 기준"에 규칙 매매 줄과 Edge AI 매매 줄이 따로 있다. 서버에 키가 없으면(구형) 규칙 매매 줄은 표시하지 않는다.
  r016 `conditions` 매도 목록에 조건 30 으로 표시.
- r001 기본값 변경(2026-11-25 사용자 결정): `EDGE_ONLY_MODE` off -> paper, `EDGE_ONLY_LOG_ONLY` True -> False (주문 없는 가상 매매가 기본으로 켜짐).

## Edge AI 실제 주문의 모델 승격 조건 + VWAP 재돌파 기록 단계 (2026-11-27)
- r001 `EDGE_LIVE_REQUIRE_PROMOTED`(bool, 기본 true, 앱에서 바꾸지 않음): Edge AI 신호(_e09)의 주문은 `runtime/edge_ai/current_meta.json`
  의 `"promoted": true` 인 모델에서만 나간다 - 병행(`EDGE_LIVE_ORDERS`)과 단독(`TRADING_STRATEGY_MODE=edge`) 모두. 미승격이면 확률이 기준을
  넘어도 사지 않고 로그 `EDGE_SIGNAL_MODEL_NOT_PROMOTED_p=.._thr=..` (종목당 10분에 한 줄). 이미 산 Edge 수량의 청산과 가상 매매(r022)는 그대로.
- r016 `set EDGE_LIVE_ORDERS=true`: 모델이 없거나 미승격이면 거부 - `rejected.EDGE_LIVE_ORDERS` =
  `"the Edge AI model is not promoted (g014) - live orders stay off; paper trading still works"` (rc 2). false 로 바꾸는 것과 한도 변경은 항상 가능.
  이미 true 로 저장된 값은 바꾸지 않는다(r003 이 신호마다 다시 판정).
- 앱 제어 탭: 모델 정보(`edge` 응답의 `promoted`)가 false / 모델 없음이면 "실제 매매" 카드에 "모델 미승격" 문구를 표시하고, 켜려 하면
  서버에 보내기 전에 안내 창으로 막는다(모델 정보를 아직 못 읽은 경우는 서버 판정을 따른다).
- r021 `postclose` 의 새 단계 `vwaplog` = `data_simulation/g055_vwap_reclaim_log.py --date D` (hi52acct 다음, scan 앞, 실패해도 작업은
  계속, collect 스위치를 따름). 주문 없음, KIS 호출 없음, r003/r004 불변. steps 에 `"vwaplog"` 추가 - 앱 단계 라벨 "VWAP 재돌파 기록(g055)".
  산출물 `<runtime>/data/vwap_reclaim/{events.csv, days.csv, report.md}`. 규칙은 g055 머리말에 고정(RULE_VERSION 1, 2026-11-27 부터가 검증 구간).
- [2026-11-28] r021 `postclose` 의 새 단계 `s1log` = `data_simulation/g061_s1_shadow_log.py --date D` (picklog 다음, apply 앞, 실패해도
  작업은 계속, scan 스위치를 따름). g002 단순 프로파일 s1 이 그날 골랐을 종목을 기록만 한다 - 선별 파일 / r004 / 주문 불변, KIS 호출 없음.
  steps 에 `"s1log"` 추가 - 앱 단계 라벨 "선정 병행 기록(g061)" ([2026-11-28] g002 기본 프로파일이 s1 이 되어, 이제 따로 기록하는 쪽은 v0). 산출물 `<runtime>/data/pick_shadow_s1/picks_D.json` (이미 있는 날짜는 다시 쓰지 않음).

## Edge AI 모델 분리: 규칙 매매 필터 모델 / Edge AI 매매 모델 (2026-11-27)
- 모델 파일이 둘이다(사용자 결정: 규칙 매매 필터는 지금처럼 유지).
  - `runtime/edge_ai/current_meta.json` - 규칙 매매 매수 필터(r005 _020) 모델. 예전과 같다(g013 기본 라벨, g014 기본 = `--role gate`).
  - `runtime/edge_ai/trade/current_meta.json` - Edge AI 매매(가상 r022, 실제 _e09) 모델. `g013 --live-label`(실제 청산값
    `EDGE_ONLY_TP_PCT` / `EDGE_ONLY_SL_PCT` / `EDGE_ONLY_HORIZON_MIN` 라벨) + `g014 --role trade`(대상 y = 그 청산의 순손익 > 0).
    이 파일이 없으면 Edge AI 매매도 앞의 파일을 읽는다(r011 `TRADE_MODEL_SUBDIR`, 1분마다 다시 판단 - 재시작 불필요).
- trade 모델 meta 의 새 필드: `"role": "trade"`, `"no_trade": bool`. 검증 구간에 "평균 순손익 > 0 이고 거래일 부트스트랩 95% 하한 > 0"
  인 기준이 없으면 `"threshold": 1.0`, `"no_trade": true` -> 어떤 확률도 기준에 닿지 않아 Edge AI 매매는 매수 0건(가상 매매 단계 표시는
  "확률 미달"). 필터 모델 meta 에는 `"role": "gate"`, `"no_trade": false`.
- r016 `set EDGE_LIVE_ORDERS=true` 의 승격 판정과 r005 `EDGE_SIGNAL_MODEL_NOT_PROMOTED` 는 Edge AI 매매가 읽는 모델(trade 가 있으면 trade)
  기준이다.
- r021 `edge_ai` 작업(과 `postclose` 의 edge_ai 부분) steps: `edge_dataset`, `edge_train`, `edge_check`, `trade_dataset`, `trade_train`
  (뒤 2개가 새 단계, 실패해도 작업은 계속). 앱 단계 라벨 "Edge AI 매매 모델 데이터셋(g013)" / "Edge AI 매매 모델 학습(g014)".
  청산값을 바꾼 뒤 첫 `trade_dataset` 은 그 라벨의 데이터셋을 처음부터 만든다(노트북 약 25분).
- 앱: 제어 탭 모델 줄은 trade 모델이 있으면 그것을 보여 준다(`edge` 읽기: `edge_ai/trade/current_meta.json` 먼저). `no_trade` 면
  "매수 기준 없음 (매수 안 함)", 가상 매매 탭의 확률 기준도 같은 문구(기준 >= 1).

## 규칙 매매 조건 목록에서 Edge AI 필터(_020) 제외 (2026-11-27)
- 사용자 결정: 11월에 규칙 매수 신호를 가장 많이 막은 필터(최근 8거래일 1,364건 중 1,260건)를 조건 체크에서 뺀다. r005 `BUY_CONDITIONS`
  에서 `_COND_020` 만 빠지고 필터 소스(`_020_edge_ai_meta_filter`, `_COND_020`, `EDGE_AI_MODE`)는 그대로다. 되돌리기: `BUY_CONDITIONS` 의
  주석 자리에 `_COND_020,` 을 다시 넣는다.
- r016 `conditions` 의 `buy` 목록에 번호 20 이 없다(1 ~ 19, 21 ~ 25, 번호는 다시 매기지 않음). 앱은 서버 목록을 그대로 그린다.
- 규칙 매매 신호의 `runtime/edge_ai/live/edge_ai_<날짜>.jsonl` 기록(필터가 남기던 것)은 더 쌓이지 않는다. r003 재시작 후 적용.

## 매매 방식 4가지: 규칙 매매(normal / extended) · Edge AI 매매 · VWAP 매매 · 반등 알림 (2026-11-27)
- 기본값: 규칙 매매 ON(`RULE_BUY_ENABLED` true, 변형 normal), Edge AI 매매 = 가상 매매 ON(`EDGE_ONLY_MODE` paper, `EDGE_LIVE_ORDERS` false),
  VWAP 매매 OFF(`VWAP_TRADE_MODE` off), 반등 알림 ON(AutoBotHtsH 앱 설정).
- 규칙 매매 변형 `RULE_TRADE_VARIANT`(str, `normal` | `extended`, r016 `set` 이 다른 값은 `must be one of normal / extended` 로 거부).
  extended = r001 `RULE_EXTENDED_VALUES` 묶음을 라이브 오버라이드 뒤에 덮어쓴다(같은 키의 저장값보다 우선, `RULE_VARIANT_APPLIED` 에 기록).
  조건 목록과 함수는 같고 값만 다르다: 1분 트리거 룩백 4봉, 트리거 유효 120초, 1분 크로스 유효 120초, 틱 크로스 유효 60초, 틱 피드 15초,
  1분 above 최대 300초, 틱봉 60 만, MA 5/15, 체결 밀도 60건/분(3분), 추격 0.5% / BB 갭 0.8%(동적 완화 끔), `TRIPLE_1MIN_BELOW_TURN_PASS` false.
  `RULE_EXTENDED_VALUES` / `RULE_VARIANT_APPLIED` 는 `set` 대상이 아니다. r016 `conditions` 의 값은 변형 적용 뒤의 값이다. r003 재시작 후 적용.
- VWAP 매매(r029_vwap_trade.py): `VWAP_TRADE_MODE`(str, `off` | `paper`, 다른 값은 거부). paper = 가상 매매(주문 없음) -
  `runtime/data/live_runtime/vwap_paper_<날짜>.jsonl`(ENTRY / EXIT, r022 의 edge_only 파일과 같은 형식, prob / threshold 는 null,
  model_id `vwap_rule_v1`) 과 `vwap_paper_status_<날짜>.json`(같은 형식, 새 stage: `below_vwap`, `no_trend`, `low_gap`(2026-11-28: 시가 갭이 `VWAP_TRADE_MIN_OPEN_GAP_PCT` 미만이거나 모름), `no_tick_cross`(`VWAP_TRADE_TICK_TRIGGER`=True 일 때만), `chase`,
  `price_drop`, `stale`). 로그 `[VWAP PAPER  ]`. 정규장 감시 종목만. 매수 4단계와 설정 키(`VWAP_TRADE_*`)는 r001 / r029 머리말.
  청산 `VWAP_TRADE_TP_PCT` / `VWAP_TRADE_SL_PCT`(%) / `VWAP_TRADE_HORIZON_MIN`(분), 남으면 15:20 청산. 실제 주문 경로는 아직 없다.
- 앱: 제어 탭 규칙 매매 카드에 Normal / Extended 선택, VWAP 매매 카드(스위치, 매수 4단계, 청산 기준 입력). 가상 매매 탭에 장부 선택
  (Edge AI 매매 / VWAP 매매) - IPC `paper:get` / `paper:status` 의 두 번째 인자 `book`(`edge` | `vwap`).

## VWAP 매매 실제 주문 (2026-11-27)
- r001 `VWAP_TRADE_LIVE_ORDERS`(bool, 기본 false). true 이고 `VWAP_TRADE_MODE=paper`, `TRADING_STRATEGY_MODE=normal`, `AUTO_SELL_MODE=all` 일 때만
  새로 산다(r003 로그 `[VWAP LIVE   ] 신규 매수 켜짐/꺼짐 (사유)`). 켜져 있는 동안 VWAP 가상 매매는 새로 사지 않는다.
  한도: `VWAP_LIVE_ORDER_KRW`(10,000..1,000,000) / `VWAP_LIVE_MAX_POSITIONS`(1..5) / `VWAP_LIVE_MAX_ENTRIES_PER_DAY`(1..20) /
  `VWAP_LIVE_DAILY_LOSS_KRW`(1,000..1,000,000) / `VWAP_LIVE_UNFILLED_CANCEL_SECONDS`(5..300). 청산값 `VWAP_TRADE_TP_PCT` / `VWAP_TRADE_SL_PCT`
  (0.1..30) / `VWAP_TRADE_HORIZON_MIN`(1..400). 범위를 벗어난 `set` 은 `out of range (lo..hi)` 로 거부.
- r016 `set` 거부 사유(`rejected`): 실제 주문이 켜진 채 `VWAP_TRADE_MODE` 를 끄거나 `AUTO_SELL_MODE` / `TRADING_STRATEGY_MODE` 를 바꾸면
  `... switch VWAP_TRADE_LIVE_ORDERS off first` (같은 `set` 에 `VWAP_TRADE_LIVE_ORDERS=false` 를 함께 보내면 통과 - 앱이 그렇게 보낸다).
- r016 `status` 에 `vwap_live`(= `edge_live` 와 같은 형식: held / open_buys / today{entries, realized, sl_codes, unpriced_sells}).
- [2026-11-28] r016 `status` 에 `last_exit` 추가: r003 이 매매 루프에 들어가기 전에 끝난 이유 `{at, reason, detail, code, pid}` 또는 null.
  `reason` = `market_closed`(휴장일) / `standby`(역할 standby) / `already_running`(다른 r003 이 잠금 보유) / `auth_failed`(KIS 인증 실패) /
  `no_watchlist`(감시목록에 종목 없음). r003 이 `live_runtime/r003_last_exit.json` 에 쓰고, 다음 실행이 매매 루프에 들어가면 지운다.
  앱: [실행] / [재시작] 뒤 봇이 바로 끝났으면 "마지막 동작"에 이유를 붉게 보여주고, 봇이 꺼져 있는 동안 봇 상태 카드에 "마지막 실행 종료" 줄을 둔다.
- 주문 / 장부: Edge AI 실제 매매와 같은 경로(r003 보조 전략). 체결 장부 `bot_fill_ledger.json` 에 `"vwap"`(종목별 수량·원가) 와
  `"vwap_day"`(그날 매수 횟수·실현 손익·손절 종목) 키가 추가된다. 주문 기록의 `strategy` 는 `"vwap"`. 매수는 정규장 KRX 매도1호가 지정가 1건,
  매도 사유 `VWAP_TP_x.xxPCT` / `VWAP_SL_x.xxPCT` / `VWAP_TIME_nMIN` / `VWAP_CLOSE_1520`. 한 종목은 한 전략만 보유한다(규칙 / Edge / VWAP).
- 매수 조건 `VWAP_BUY_CONDITIONS`(r005): 공통 상태 조건 1~8, `_v09_vwap_signal`(가상 매매와 같은 r029.vwap_signal, 당일 VWAP 은 일중 기준가
  조회값), 23~25.
- 앱: 제어 탭 VWAP 매매 카드 "실행 대상"에 실제 매매 체크(경고 2회) / 가상 매매 표시, 실제 주문 한도 입력과 오늘 현황.

## 규칙 매매 매수 조건별 적용 여부 (2026-11-27, Codex 검토 4회)
- r001 `BUY_CONDITION_OFF`(list[int], 기본 `[20]`): 규칙 매매(normal / basic)가 적용하지 않는 매수 조건 번호. `BUY_CONDITIONS` 에는 1~25 가
  모두 있고(20 복귀), 실제 건너뛰기는 r005 `check_buy_conditions` 가 한다. Edge AI 매매 / VWAP 매매의 자체 목록에는 적용하지 않는다.
- 끌 수 없는 조건(r005 `LOCKED_BUY_CONDITIONS`): 1, 3, 4, 5, 6, 9, 11, 14, 19, 23, 24, 25. 파일에 직접 넣어도 실행 시 무시하고 r003 이 경고한다.
  2, 15 는 꺼도 뒤 조건이 읽는 값(세션 시작 시각 / 전일 종가·상승률)만 계산하고 조건 함수는 부르지 않는다.
- r016 `set BUY_CONDITION_OFF=[..]`: 없는 번호 / 끌 수 없는 번호가 있으면 거부(`unknown condition number(s)` / `condition(s) that cannot be
  switched off: ...`). 앱은 목록 전체를 보내지 않고 `set BUY_CONDITION_SWITCH={"no": N, "apply": true|false}` 를 쓴다 - r016 이 쓰기 잠금
  안에서 지금 저장된 목록에 번호 하나만 넣고 뺀다(응답 `accepted.BUY_CONDITION_OFF` = 새 목록, 거부는 `rejected.BUY_CONDITION_SWITCH`).
- r016 `conditions` 의 `buy` 각 행에 추가: `apply`("on" 차단 가능 / "off" 아무것도 막지 않음 / "record" 평가·기록만),
  `off_by`(["BUY_CONDITION_OFF", "ENABLE_X=False", ...] - 조건 전체 해제와 조건 안 설정값을 구분), `switchable`(bool), `locked_reason`,
  `prep`(bool), `apply_running`(실행 중인 r003 의 상태, 조건 함수 이름으로 연결. null = 꺼짐 / 이전 버전). 문서에 `buy_condition_off`,
  `buy_condition_off_ignored`. `apply` 는 저장된 설정(다음 시작 기준)이다. `buy` 목록은 꺼진 조건도 포함한다(1~25).
- r003 `r003_running.json` 에 `buy_condition_state`({조건 함수 이름: on|off|record}). 시작 로그 `[R003 MODE] BUY_CONDITION_OFF=[..]`.
- 앱: 규칙 매매 세부 설정 목록에 적용 / 미적용 / 기록만 / 필수 / 재시작 대기 표시, 조건별 "조건 전체 사용" 체크. 파이프라인은 오늘(실시간)
  화면에서만 미적용 / 기록만 표시. 조건 안의 `ENABLE_*` 는 "옵션 n/m" 으로 따로 표시한다.
- 과거 데이터 재생(g003)은 r005 목록을 쓰지 않으므로 이 설정이 자동으로 반영되지 않는다.

## 보조 전략(Edge AI / VWAP)의 봉별 판정 기록과 타임라인 매매 방식 탭 (2026-11-27, Codex 검토 반영)

- 새 파일 `runtime/data/live_runtime/strategy_eval_<strategy>_<date>.jsonl` (`strategy` = `edge` | `vwap`, r030_strategy_trace.py).
  보조 전략이 1분봉 하나를 판정할 때마다 한 줄: `{"t": 판정 시각, "bar": "HH:MM", "code", "mode": "live"|"paper", "stage", ...}`.
  `stage`: VWAP `below_vwap / no_trend / low_gap / no_tick_cross / chase / price_drop / stale / no_history`, Edge `below_mid / no_trigger /
  below_threshold / not_promoted / no_model`, 공통 `cond_before / cond_after`(공통 안전 조건, `no` = 조건 번호, `cond`, `detail`) `/ signal /
  ordered / order_refused / bought / candidate / held / sold / wait_bars / outside_window / limit / fund / sl_today / error`.
  그 밖의 키(`vwap`, `gap`, `tick`, `tick_cross`, `bar_age`, `prob`, `thr`, `price`, `reason`, `qty`)는 봇이 그 판정에 쓴 값 그대로다.
- 기록 전용이다: 판정 / 주문 / 상태에 쓰이지 않고 봇이 다시 읽지 않는다. 호출은 큐에 넣기만 하고 파일 쓰기는 별도 스레드가 한다(큐가
  차면 버리고 센다). 실주문 경로는 r003 `_side_trace`(`_edge_live_try_buy` 의 판정 뒤 / 주문 뒤), 가상 경로는 r022 `_stage`(새 봉의 첫 단계
  호출, 감시 종목만). 가상 매매 실행기는 r003 / r029.make_runner 가 켠 것만 기록한다.
- r003: 보조 전략 매수가 계좌 반영보다 먼저 체결 장부로 종결될 때 `<date>_buy_sell.log` 에 `>>> <code>_<name> Buy - ...` 블록이 빠지던 것을
  기록한다(가격 = 그 주문의 lot 평균, 모르면 N/A). 체결 반영 / 종결 로그의 태그는 주문의 전략(`[VWAP LIVE]` / `[EDGE LIVE]`)을 쓴다.
- 앱: 타임라인에 매매 방식 탭(자동 / 규칙 / Edge AI / VWAP). 보조 전략 탭은 위 파일과 주 로그의 `[VWAP LIVE]` `[EDGE LIVE]` `[VWAP PAPER]`
  `[EDGE ONLY]` 줄(신호 / 주문 / 매수 안 함 / 보유 점검 / 청산 / 신규 매수 중단)을 그대로 보여 준다. 판정을 다시 계산하지 않는다.
  블록이 없는 날(2026-11-27 이전 봇)의 보조 전략 매수는 `[BUY SUBMIT]` / `[BUY PENDING]` 줄을 주문번호로 이어 거래 목록에 넣는다(가격은
  주문가, `≈` 표시). main IPC `strategy:evals(date, strategy, code)`.
- 매매 방식 현황 (2026-11-27, 앱 전용 · 서버 변경 없음): 탭 `method` (AutoBotHtsH 창 `bot-method`) - 방식별 감시 · 기록 종목 표
  (`shared/methodBoard.ts`: 감시 종목 + 가상 러너 상태의 수집기 종목 + 그 방식의 기록 / 체결이 있는 종목, 종목별 마지막 기록은
  봉별 판정(r030) / 봇 로그 / 가상 러너 상태 중 가장 최근 것). main IPC `strategy:evalLatest(date, strategy, upTo)` = 종목별 마지막
  판정 행 + 행 수. 켜기 / 끄기 · 설정은 제어 탭 그대로(r016).
- 직접 매도 기록 (2026-11-27, r003 재시작 후): 보조 전략이 산 수량이 직접 매도로 빠지면 r003 이 메인 로그와 거래 로그에
  `[VWAP LIVE   ] <code>_<name> | 직접 매도 N주 기록 | 매도가=<가격|N/A> 원가=<가격|N/A> 손익=<±원(±%)|N/A> | strategy=<edge|vwap> order=<키> | ...`
  한 줄을 쓴다(기록 전용 - 전략 실현 손익 / 한도 불변, `r023.hand_sale_notes`). 앱은 거래 로그의 이 줄을 그 방식의 매도(`Trade.hand`)로
  표시한다(가격이 N/A 면 결과 행 없음).
- 매매법별 대상 종목 (2026-11-27, r003 재시작 후 적용): r001 `RULE_TARGET_CODES` / `EDGE_TARGET_CODES` / `VWAP_TARGET_CODES` (종목코드
  6자리 문자열 목록, 빈 목록 = 지금까지의 범위 전체). 목록이 있으면 그 매매법은 목록의 종목만 새로 산다(좁히기만 함, 보유분 매도 /
  청산 / 미체결 관리는 그대로). `r016 set` 은 코드 형식 · 개수(300)만 검사하고 정렬 · 중복 제거해 저장한다(감시목록 포함 여부는
  검사하지 않음). `r016 get` 의 `values[KEY].effective / running / pending` 그대로. 읽을 수 없는 값이면 그 매매법은 새로 사지 않는다.
  로그: `[R003 MODE   ] <method> 대상 종목 N개만 신규 매수: ...`, `[TARGET      ] <sym> | <method> 대상 종목 아님 - 신규 매수 평가 안 함`
  (종목 · 방식당 30분에 한 번). 가상 러너 상태 단계 `not_target`.
  대상 목록이 하나라도 있으면, 어느 방식도 새로 살 수 없는 종목은 활성 감시에서 빠진다: `[ACT DROPOUT ] <sym> | reason=NOT_TARGET`
  (대기열 맨 뒤로, 다시 후보가 되면 승격).
- 대상 종목 자동 구성 (2026-11-27, `r031_method_targets.py`, 설정 `<runtime>/config/method_targets.json`): A = 그날 g002 picks,
  규칙 = A + 스캔 순위(`_D_scan_all.md`) 다음 종목으로 `sizes.rule`(30)까지, Edge AI = A + 그다음 순위로 `sizes.edge`(50)까지,
  VWAP = 합집합. 방식별 `include` / `exclude`(항상 포함 / 제외)는 날마다 유지. 감시목록(r004) = 합집합(A, 규칙 추가, Edge 추가 순).
  - `r016 targets get [--date D]` -> `{ok, date, config, config_state, composition{a, rule_extra, edge_extra, lists, watch, rows{code:
    {name, src, rank, score, eligible, fail, in, included}}, tick_cap, no_tick, dropped, short, excluded, digest}, stored, watch_now,
    active_size}` (읽기 전용).
  - `r016 targets config --stdin` (JSON: `auto`, `sizes`, `exclude_funds`, `include` / `exclude` 전체 교체, 또는 한 종목씩
    `include_add` / `include_remove` / `exclude_add` / `exclude_remove` `{method: [codes]}`).
  - `r016 targets apply [--date D] [--expect DIGEST] [--restart-collector]`: 세 대상 목록을 먼저 저장하고 r004 를 쓴다
    (r003 재시작 후 적용). `--expect` 가 현재 구성의 digest 와 다르면 거부.
  - `auto: true` 이면 장 마감 작업의 `watchlist apply --date D` 가 picks 복사 대신 이 구성을 적용한다(기본 false = 기존 동작).
    `watchlist_vs_picks` 의 `applied` 는 구성 적용 시 "picks 가 모두 감시목록에 있음"으로 판정(`composed: true`).
- 매매 방식 현황 대시보드 (2026-11-27, 앱): 탭 `method` 의 값 `all | rule | edge | vwap`. 실행 단위 5개(규칙 / Edge AI 가상 · 실전 /
  VWAP 가상 · 실전)의 저장 스위치와 실행 상태는 `shared/methodUnits.ts` (`unitInfos`: r016 status.running + get 의 running 값이
  있을 때만 "실행 중").
