# 네이버증권에서 종목코드로 종목명을 실시간 조회하여 "종목코드,종목명" 형식으로 출력합니다.
# requirements: requests, beautifulsoup4

import os
import requests
from bs4 import BeautifulSoup
import time

# 입력 파일: r009_universe_symbols_master.txt (스크립트와 같은 폴더)
input_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "r009_universe_symbols_master.txt")

# 네이버증권 종목명 조회 함수
def get_stock_name_naver(code):
    url = f"https://finance.naver.com/item/main.nhn?code={code.zfill(6)}"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        name_tag = soup.select_one(".wrap_company h2")
        if name_tag:
            return name_tag.text.strip().split()[0]
        else:
            return None
    except Exception as e:
        return None

if __name__ == "__main__":
    try:
        with open(input_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) < 1:
                    continue
                code = parts[0].strip()
                if not code:
                    continue
                name = get_stock_name_naver(code)
                if name:
                    print(f"{code},{name}")
                else:
                    print(f"{code},[조회실패]")
                time.sleep(0.2)  # 네이버 서버에 부담을 주지 않도록 약간의 딜레이
    except FileNotFoundError:
        print(f"[ERROR] 입력 파일을 찾을 수 없습니다: {input_file}")
        print("스크립트와 같은 폴더에 r009_universe_symbols_master.txt 파일이 있는지 확인하세요.")
# r009_print_code_name_pairs.py
# 이 스크립트는 r009_universe_symbols_master.txt 파일에서 각 줄의 종목코드와 종목명을 추출하여 "종목코드,종목명" 형식으로 출력합니다.


import os

# 현재 스크립트 파일과 같은 디렉토리에 있는 r009_universe_symbols_master.txt를 찾음
input_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "r009_universe_symbols_master.txt")

try:
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):  # 빈 줄 또는 주석 무시
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            code = parts[0].strip()
            name = parts[1].strip()
            print(f"{code},{name}")
except FileNotFoundError:
    print(f"[ERROR] 입력 파일을 찾을 수 없습니다: {input_file}")
    print("스크립트와 같은 폴더에 r009_universe_symbols_master.txt 파일이 있는지 확인하세요.")
