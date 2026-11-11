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
| `doctor` | `{ok, problems[], host, python, executable, packages{}, code{dir,commit}, open_trading_api{path,exists,commit}, kis_config{dir,devlp_yaml,token_today}, runtime{root,exists}}` | 0 |
| `job list` | `{ok, running: {pid, started_at, name, run_id}\|null, jobs: [{name, title, steps, last}]}` | 0 |
| `job start NAME [--date D] [--workers N] [--edge-ai] [--keep-watchlist] [--trading-day-only]` | `{ok, name, pid, run_id, running}` / 휴장일 `{ok, skipped, reason}` / 다른 작업 실행 중 `{ok:false, error, running}` | 0 / 3 |
| `job stop NAME` | `{ok, stopped}` | 0 / 3 |
| `job log NAME [-n N]` | `{ok, name, log, lines[], state}` | 0 |
| `job data --date D` | `{ok, date, dir, exists, scannable, collected, collected_count, files_10s, files_daily, data_mtime, picks: [{code,name}]\|null, picks_file, picks_mtime, scan_log_mtime, r004: {scan_date, same_codes, applied, count, mtime}}` - [2026-11-11] 그 날짜 폴더의 수집 데이터 유무와 g002 선별 결과, r004 가 그 선별과 같은지 | 0 / 2 |
| `watchlist apply --date D` | `watchlist set` 과 같은 `{ok, count, backup, path, restart_required, collector_missing, collector_note}` + `date, picks_file, for_trading_day` / picks 파일 없음 `{ok:false, error}` - [2026-11-11] 그 날짜의 `_D_picks.txt` 를 r004 로 (헤더 scan_date / for_trading_day) | 0 / 2 |

- 작업(`live_trading/r021_jobs.py`): `collect`(g001 --date D), `history`(g025 --append D), `scan`(g002 --date D --export-watchlist
  --for-next-trading-day; `--keep-watchlist`면 g002 --date D --no-export-watchlist: 그 날짜의 선별 파일만 만들고 r004는 그대로 - 과거 날짜
  보충 수집/Edge AI 재학습용, 앱의 "감시목록(r004) 갱신" 체크 해제; [2026-11-11] 앱의 "감시 종목 선별" 카드는 기본으로 이 모드로 돌린 뒤
  `job data` 로 읽은 선별 결과를 보여주고 "r004 감시목록에 적용" 체크로 `watchlist apply` 를 부른다), `edge_ai`(g013 --live, g014, g024 --edge-ai), `universe`(g010), `postclose`(collect -> history -> scan
  [-> edge_ai]). collect/scan 실패 시 뒤 단계 생략, history 실패는 계속. 한 번에 한 작업. 상태 `runtime/data/jobs/<NAME>.json`,
  로그 `runtime/log/jobs/<YYYYMMDD>/<NAME>_<HHMMSS>.log`.
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
