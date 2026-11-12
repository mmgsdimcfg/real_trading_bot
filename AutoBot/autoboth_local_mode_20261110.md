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
| `job start NAME [--date D] [--workers N] [--edge-ai] [--no-collect] [--no-scan] [--no-apply] [--email] [--scheduled] [--trading-day-only]` | `{ok, name, pid, run_id, running}` / 휴장일 `{ok, skipped, reason}` / 다른 작업 실행 중 `{ok:false, error, running}` | 0 / 3 |
| `job stop NAME` | `{ok, stopped}` | 0 / 3 |
| `job log NAME [-n N]` | `{ok, name, log, lines[], state}` | 0 |
| `job data --date D` | `{ok, date, dir, exists, scannable, collected, collected_count, files_10s, files_daily, data_mtime, picks: [{code,name}]\|null, picks_file, picks_mtime, scan_log_mtime, r004: {scan_date, same_codes, applied, count, mtime}}` - [2026-11-11] 그 날짜 폴더의 수집 데이터 유무와 g002 선별 결과, r004 가 그 선별과 같은지 | 0 / 2 |
| `watchlist apply --date D [--restart-collector]` | `watchlist set` 과 같은 `{ok, count, backup, path, restart_required, collector_missing, collector_note}` + `date, picks_file, for_trading_day` (+ `collector_restart` when --restart-collector and a running g011 misses new codes) / picks 파일 없음 `{ok:false, error}` - [2026-11-11] 그 날짜의 `_D_picks.txt` 를 r004 로 (헤더 scan_date / for_trading_day) | 0 / 2 |

- 작업(`live_trading/r021_jobs.py`): `collect`(g001 --date D), `history`(g025 --append D), `scan`(g002 --date D --no-export-watchlist ->
  `apply`), `apply`(r016 watchlist apply --date D: 그 날짜의 `_D_picks.txt` 를 r004 로 - 백업/이력/다음 거래일 헤더), `edge_ai`(g013 --live,
  g014, g024 --edge-ai), `universe`(g010), `postclose`(collect -> history -> scan -> apply [-> edge_ai]). [2026-11-11] 단계 선택:
  `--no-collect`(수집+이력 생략, 그 날짜에 10초봉 파일이 있어야 함), `--no-scan`(선별 생략, 선별 파일이 있어야 함), `--no-apply`(r004 반영
  생략; `--keep-watchlist` 는 옛 이름). 앞 단계 산출물이 없으면 start 가 `{ok:false, error}` 로 거부. 앱의 장 마감 파이프라인 카드는
  ① 수집 ② 선별 ③ r004 반영 체크로 이 스위치를 만든다(1+2+3 / 2+3 / 3). apply 는 선별 파일이 없거나 이 실행의 scan 보다 오래되면
  건너뛴다(step state=skipped + note, 작업은 done; 상태 JSON `note`). collect/scan/apply 실패 시 뒤 단계 생략, history 실패는
  계속. [2026-11-11] `--email`(postclose/scan): scan 단계를 `g002 --notify-email` 로 실행해 선별 종목(본문)과 리포트 파일(첨부)을
  메일로 보낸다(g016, 실행 호스트의 `~/.mail/mail_config.json`; scan 단계가 없으면 보내지 않음, 전송 실패는 작업 실패가 아님 -
  로그의 `[EMAIL]` 줄). 앱은 파이프라인 카드의 "선별 결과 메일" 체크와 설정 `postcloseEmail`(자동 실행)로 붙인다.
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
