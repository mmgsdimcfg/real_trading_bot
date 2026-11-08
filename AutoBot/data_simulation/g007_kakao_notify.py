"""
g007_kakao_notify.py

카카오톡 "나에게 보내기"(talk_message, 기본 메모 API)로 g002 스캔 결과를
내 카카오톡으로 전송한다. 참고: https://developers.kakao.com/docs/latest/ko/kakaotalk-message/rest-api

최초 1회 설정
-------------
1. https://developers.kakao.com 에서 앱 생성 후:
   - 앱 생성 화면의 "앱 대표 도메인"에는 경로(path) 없이 프로토콜+호스트+포트만 입력
     (예: http://localhost:5000 - "/oauth" 같은 경로를 붙이면 "유효하지 않은 URL"
     오류가 난다). 실제 서비스 도메인이 아니어도 개인 테스트 용도로는 무방.
   - 앱 설정 > 플랫폼 > Web에도 동일하게 http://localhost:5000 (경로 없이) 등록
   - "카카오 로그인" 활성화 후 "Redirect URI"에는 경로를 포함한 전체 주소를 등록
     (예: http://localhost:5000/oauth - 이 필드는 path 포함 가능하고 localhost도 허용됨.
     실제로 서버를 띄우지 않고 인가코드만 복사해 쓸 것이므로 응답을 받는 서버가 없어도 무방)
   - "카카오 로그인 > 동의항목"에서 `talk_message`("카카오톡 메시지 전송") 를 사용 설정
   - "보안"에서 Client Secret을 활성화했다면 값을 기록해둔다 (미활성화면 비워둠)
2. ~/.kakao/kakao_config.json 파일을 아래 형식으로 만든다 (git에 올리지 말 것):
   {
     "rest_api_key": "REST API 키",
     "client_secret": "",
     "redirect_uri": "http://localhost:5000/oauth"
   }
3. python g007_kakao_notify.py --authorize
   -> 출력된 URL을 브라우저에서 열어 로그인/동의 -> redirect_uri로 이동 -> 주소창의
      "code=..." 뒤 값을 복사해 프롬프트에 입력하면 access_token/refresh_token이
      ~/.kakao/kakao_token.json 에 저장된다.
4. python g007_kakao_notify.py --test  로 정상 전송 확인.

이후에는 send_text()/send_watchlist() 호출 시 만료 임박한 access_token을
refresh_token으로 자동 갱신한다. refresh_token 자체가 만료(장기 미사용)되면
--authorize를 다시 실행해야 한다.
"""

from __future__ import annotations

import argparse
import json
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlencode

import requests

CONFIG_DIR = Path.home() / ".kakao"
CONFIG_PATH = CONFIG_DIR / "kakao_config.json"
TOKEN_PATH = CONFIG_DIR / "kakao_token.json"

AUTHORIZE_URL = "https://kauth.kakao.com/oauth/authorize"
TOKEN_URL = "https://kauth.kakao.com/oauth/token"
SEND_URL = "https://kapi.kakao.com/v2/api/talk/memo/default/send"

# 기본 텍스트 템플릿은 글자수 제한이 있으므로 여유를 두고 컷한다.
TEXT_TEMPLATE_MAX_CHARS = 190


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise SystemExit(
            f"{CONFIG_PATH} 가 없습니다. rest_api_key/redirect_uri(및 필요시 client_secret)를 "
            f"담은 JSON 파일을 먼저 만들어주세요. (모듈 docstring 참고)"
        )
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _load_token() -> dict | None:
    if not TOKEN_PATH.exists():
        return None
    return json.loads(TOKEN_PATH.read_text(encoding="utf-8"))


def _save_token(token: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    token = dict(token)
    token["obtained_at"] = time.time()
    TOKEN_PATH.write_text(json.dumps(token, ensure_ascii=False, indent=2), encoding="utf-8")
    TOKEN_PATH.chmod(0o600)


def authorize() -> None:
    """최초 1회: 브라우저 인가 -> 인가코드 -> access/refresh 토큰 발급."""
    config = _load_config()
    params = {
        "client_id": config["rest_api_key"],
        "redirect_uri": config["redirect_uri"],
        "response_type": "code",
        "scope": "talk_message",
    }
    url = f"{AUTHORIZE_URL}?{urlencode(params)}"
    print(f"브라우저에서 아래 URL을 열어 로그인/동의하세요:\n{url}\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    code = input("리다이렉트된 주소의 'code=' 뒤 값을 붙여넣으세요: ").strip()

    data = {
        "grant_type": "authorization_code",
        "client_id": config["rest_api_key"],
        "redirect_uri": config["redirect_uri"],
        "code": code,
    }
    if config.get("client_secret"):
        data["client_secret"] = config["client_secret"]

    resp = requests.post(TOKEN_URL, data=data, timeout=10)
    resp.raise_for_status()
    _save_token(resp.json())
    print(f"토큰 저장 완료: {TOKEN_PATH}")


def _refresh_access_token(config: dict, token: dict) -> dict:
    data = {
        "grant_type": "refresh_token",
        "client_id": config["rest_api_key"],
        "refresh_token": token["refresh_token"],
    }
    if config.get("client_secret"):
        data["client_secret"] = config["client_secret"]

    resp = requests.post(TOKEN_URL, data=data, timeout=10)
    resp.raise_for_status()
    new_token = resp.json()
    # 카카오는 refresh_token을 매번 재발급하지 않을 수 있으므로 기존 값 위에 덮어쓴다.
    merged = {**token, **new_token}
    _save_token(merged)
    return merged


def _get_valid_access_token() -> str:
    config = _load_config()
    token = _load_token()
    if token is None:
        raise SystemExit(
            "저장된 카카오 토큰이 없습니다. 먼저 `python g007_kakao_notify.py --authorize` 를 실행하세요."
        )
    obtained_at = token.get("obtained_at", 0)
    expires_in = token.get("expires_in", 0)
    if time.time() >= obtained_at + expires_in - 60:
        token = _refresh_access_token(config, token)
    return token["access_token"]


def send_text(text: str, link_url: str | None = None) -> bool:
    """카카오톡 '나에게 보내기'로 기본 텍스트 템플릿을 전송한다. 성공 시 True."""
    try:
        access_token = _get_valid_access_token()
    except Exception as exc:
        print(f"[KAKAO] 토큰 조회/갱신 실패: {exc}")
        return False

    template_object = {
        "object_type": "text",
        "text": text[:TEXT_TEMPLATE_MAX_CHARS],
        "link": {
            "web_url": link_url or "https://developers.kakao.com",
            "mobile_web_url": link_url or "https://developers.kakao.com",
        },
    }
    try:
        resp = requests.post(
            SEND_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            data={"template_object": json.dumps(template_object, ensure_ascii=False)},
            timeout=10,
        )
    except Exception as exc:
        print(f"[KAKAO] 전송 요청 실패: {exc}")
        return False

    if resp.status_code != 200:
        print(f"[KAKAO] 전송 실패 {resp.status_code}: {resp.text}")
        return False
    return True


def send_watchlist(picks: list[str], label: str, max_items: int = 20) -> bool:
    """g002 스캔 결과(picks: ['코드_종목명', ...])를 정리해서 전송한다."""
    if not picks:
        body = f"[{label}] 감시종목 스캔 결과: 선정 없음"
    else:
        shown = picks[:max_items]
        lines = [f"[{label}] 감시종목 {len(picks)}종목"]
        lines.extend(f"- {p}" for p in shown)
        if len(picks) > max_items:
            lines.append(f"...외 {len(picks) - max_items}종목")
        body = "\n".join(lines)
    return send_text(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kakao '나에게 보내기' notifier")
    parser.add_argument("--authorize", action="store_true", help="최초 1회 토큰 발급")
    parser.add_argument("--test", action="store_true", help="테스트 메시지 전송")
    args = parser.parse_args()

    if args.authorize:
        authorize()
    elif args.test:
        ok = send_text("[xgraph] 카카오톡 나에게 보내기 테스트 메시지입니다.")
        print("전송 성공" if ok else "전송 실패")
    else:
        parser.print_help()
