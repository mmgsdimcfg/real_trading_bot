# autobot-r003 systemd user service

screen 세션 대신 systemd --user 로 r003 을 관리한다. AutoBotH / r016_control.py 의 start/stop/restart 가 이 유닛을 부른다.

설치 (서버에서 1회):
```
cp /data/srv/autobot/real_trading_bot/AutoBot/live_trading/systemd/autobot-r003.service ~/.config/systemd/user/
systemctl --user daemon-reload
loginctl show-user $USER -p Linger      # Linger=yes 여야 로그아웃 후에도 유지 (이미 yes)
```
사용:
```
systemctl --user start autobot-r003     # screen 의 r003 가 돌고 있으면 먼저 종료 (r016 start 는 자동 거부)
systemctl --user status autobot-r003
journalctl --user -u autobot-r003 -f
```
호스트 역할: 노트북(윈도우, AutoBotH 로컬 모드)이 메인일 때 라즈베리파이는 `<runtime>/config/host_role.json` = standby 이다.
standby 동안 r003 은 시작을 거부하고(rc 3 - 유닛의 RestartPreventExitStatus=3 으로 재시작하지 않음, 2026-11-10 유닛 갱신 후 다시 설치) g022 cron 은 g011 을 띄우지 않는다.
라즈베리파이로 매매하려면 먼저 `python live_trading/r016_control.py role set primary` (또는 앱 제어 탭). 자세한 절차:
`AutoBot/autoboth_local_mode_20261110.md`. 윈도우에는 systemd 가 없어 `live_trading/r020_supervisor.py` 가 같은 역할을 한다.

옵션은 `<runtime>/config/r003.env` (r016 `start --dry-run --profile neutral` 가 씀). 부팅 시 자동 시작은 켜지 않는다
(`systemctl --user enable autobot-r003` 로 켤 수 있음).
