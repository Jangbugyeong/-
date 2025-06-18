from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

# ChromeDriver 자동 설치 및 브라우저 실행
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# 국민신문고 제안 페이지 접속
driver.get("https://www.epeople.go.kr/nep/prpsl/opnPrpl/opnpblPrpslList.npaid")
print(driver.title)  # 페이지 타이틀 출력

import time
time.sleep(3)

# 게시글 테이블 전체 선택
table = driver.find_element(By.CSS_SELECTOR, "table.tbl.default.brd1")
rows = table.find_elements(By.CSS_SELECTOR, "tr")  # 모든 행 가져오기

data_list = []  # 수집된 데이터 저장용 리스트

# 각 행 반복
for idx, row in enumerate(rows):
    try:
        cols = row.find_elements(By.TAG_NAME, "td")
        if len(cols) < 4:
            print(f"[{idx}] 컬럼이 부족하여 건너 뜀")
            continue

        # 글 정보 추출
        doc_num = cols[0].text.strip()
        title = cols[1].text.strip()
        dept = cols[2].text.strip()
        date = cols[3].text.strip()
        print(f"[{idx}] 번호 {doc_num}, 상세 페이지 이동 중")

        # 제목 클릭 → 상세 페이지 진입
        title_link = cols[1].find_element(By.TAG_NAME, "a")
        title_link.click()
        time.sleep(2)

        # 본문 추출
        contents = driver.find_elements(By.CSS_SELECTOR, ".b_cont")
        full_text = "\n\n".join([c.text.strip() for c in contents])

        # 딕셔너리 형태로 저장
        item = {
            "num": doc_num,
            "title": title,
            "dept": dept,
            "date": date,
            "content": full_text
        }
        data_list.append(item)

        print(f"[{idx}] 저장 완료 -> 목록 복귀")

        # 다시 목록으로 돌아가기
        driver.back()
        time.sleep(2)

    except Exception as e:
        print(f"[{idx}] 에러 발생 {e}")

# 수집 결과 JSON 파일로 저장
import json
with open("epeople_articles.json", "w", encoding="utf-8") as f:
    json.dump(data_list, f, ensure_ascii=False, indent=2)

# 브라우저 종료
driver.quit()
print(f"\n총 {len(data_list)}건 저장 완료")