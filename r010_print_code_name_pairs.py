# 네이버증권에서 종목코드로 종목명 및 시장구분(KOSPI/KOSDAQ)을 실시간 조회하여 
# "종목코드,종목명,시장구분" 형식으로 출력합니다.
# requirements: requests, beautifulsoup4

import os
import requests
from bs4 import BeautifulSoup
import time

# 입력 파일: r009_universe_symbols_master.txt (스크립트와 같은 폴더)
current_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(current_dir, "r009_universe_symbols_master.txt")

# 네이버증권 종목명 및 시장구분 조회 함수
def get_stock_info_naver(code):
    # 코드가 숫자형태일 때를 대비해 6자리 문자열로 패딩
    code_str = str(code).zfill(6)
    url = f"https://finance.naver.com/item/main.nhn?code={code_str}"
    
    # 네이버 차단 방지를 위한 브라우저 헤더 추가
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # 1. 종목명 추출
        name_tag = soup.select_one(".wrap_company h2")
        stock_name = None
        if name_tag:
            stock_name = name_tag.text.strip().split()[0]
        else:
            return None, "UNKNOWN"
            
        # 2. 시장 구분(KOSPI / KOSDAQ) 판별
        market_type = "UNKNOWN"
        
        # 메타 태그(og:description) 속성 분석
        meta_desc = soup.find("meta", {"property": "og:description"})
        if meta_desc and meta_desc.get("content"):
            desc_content = meta_desc["content"]
            if "코스피" in desc_content:
                market_type = "KOSPI"
            elif "코스닥" in desc_content:
                market_type = "KOSDAQ"
                
        # 메타 태그로 확인이 안 될 경우, 상단 이미지 클래스로 교차 검증
        if market_type == "UNKNOWN":
            img_tag = soup.select_one(".wrap_company img")
            if img_tag and img_tag.get("class"):
                classes = img_tag.get("class")
                if "kospi" in classes:
                    market_type = "KOSPI"
                elif "kosdaq" in classes:
                    market_type = "KOSDAQ"
                    
        return stock_name, market_type

    except Exception as e:
        return None, "UNKNOWN"

if __name__ == "__main__":
    try:
        with open(input_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                
                # 빈 줄, 주석(#), 또는 CSV 헤더 행(code,name) 건너뛰기
                if not line or line.startswith("#") or line.startswith("code,"):
                    continue
                    
                parts = line.split(",")
                if len(parts) < 1:
                    continue
                    
                code = parts[0].strip()
                if not code:
                    continue
                
                # 네이버에서 종목명과 시장 구분 실시간 조회
                name, market = get_stock_info_naver(code)
                
                if name:
                    # 결과 출력 (종목코드,종목명,시장구분)
                    print(f"{code},{name},{market}")
                else:
                    print(f"{code},[조회실패],UNKNOWN")
                
                # 네이버 서버 부담 방지 및 안정적인 조회를 위한 딜레이 (0.3초 추천)
                time.sleep(0.3)
                
    except FileNotFoundError:
        print(f"[ERROR] 입력 파일을 찾을 수 없습니다: {input_file}")
        print("스크립트와 같은 폴더에 r009_universe_symbols_master.txt 파일이 있는지 확인하세요.")