# autobot-r003 systemd user service

screen 세션 대신 systemd --user 로 r003 을 관리한다. AutoBotH / r016_control.py 의 start/stop/restart 가 이 유닛을 부른다.

설치 (서버에서 1회):
```
cp /data/srv/autobot/auto_trading_src/live_trading/systemd/autobot-r003.service ~/.config/systemd/user/
systemctl --user daemon-reload
loginctl show-user $USER -p Linger      # Linger=yes 여야 로그아웃 후에도 유지 (이미 yes)
```
사용:
```
systemctl --user start autobot-r003     # screen 의 r003 가 돌고 있으면 먼저 종료 (r016 start 는 자동 거부)
systemctl --user status autobot-r003
journalctl --user -u autobot-r003 -f
```
옵션은 `<runtime>/config/r003.env` (r016 `start --dry-run --profile neutral` 가 씀). 부팅 시 자동 시작은 켜지 않는다
(`systemctl --user enable autobot-r003` 로 켤 수 있음).
