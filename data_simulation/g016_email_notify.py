"""
g016_email_notify.py

g002 스캔 결과를 메일로 전송한다(종목 리스트는 본문, 리포트 파일은 첨부).
g002 --notify-email 로 스캔 마지막에 호출된다. Python 표준 smtplib만 사용.

최초 1회 설정
-------------
1. Gmail 계정에서 2단계 인증을 켜고 "앱 비밀번호"(16자리)를 발급한다.
2. ~/.mail/mail_config.json 파일을 아래 형식으로 만든다 (git에 올리지 말 것, chmod 600):
   {
     "smtp_host": "smtp.gmail.com",
     "smtp_port": 587,
     "user": "보내는계정@gmail.com",
     "password": "앱 비밀번호",
     "to": ["받는주소1", "받는주소2"]
   }
   "to"는 문자열 하나("a@x.com, b@y.com")도 가능.
3. python g016_email_notify.py --test  로 정상 전송 확인.
"""

from __future__ import annotations

import argparse
import json
import smtplib
from email.message import EmailMessage
from pathlib import Path

CONFIG_PATH = Path.home() / ".mail" / "mail_config.json"


def send_scan_report(picks: list[str], label: str, attachments: list[Path]) -> bool:
    """picks(['코드_종목명', ...])를 본문에, attachments 파일을 첨부해 전송한다. 실패해도 예외 없이 False."""
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        msg = EmailMessage()
        msg["Subject"] = f"[g002] {label} 감시종목 {len(picks)}종목"
        msg["From"], msg["To"] = cfg["user"], cfg["to"]
        msg.set_content("\n".join(picks) if picks else "선정 없음")
        for p in attachments:
            if p.exists():
                msg.add_attachment(p.read_bytes(), maintype="text", subtype="plain", filename=p.name)
        with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"], timeout=30) as s:
            s.starttls()
            s.login(cfg["user"], cfg["password"])
            s.send_message(msg)
        return True
    except Exception as exc:
        print(f"[EMAIL] 전송 실패: {exc}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="g002 scan result email notifier")
    parser.add_argument("--test", action="store_true", help="테스트 메일 전송")
    args = parser.parse_args()

    if args.test:
        ok = send_scan_report(["000000_테스트종목"], "TEST", [])
        print("전송 성공" if ok else "전송 실패")
    else:
        parser.print_help()
