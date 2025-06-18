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

# 페이지 로딩 대기
import time
time.sleep(3)

# 첫 번째 게시글 추출
row = driver.find_element(By.CSS_SELECTOR, "tbody tr")  # 표의 첫 번째 행
cols = row.find_elements(By.TAG_NAME, "td")             # 그 행의 열(td)

# 각 열에서 텍스트 추출
doc_num = cols[0].text.strip()   # 문서 번호
title = cols[1].text.strip()     # 제목
dept = cols[2].text.strip()      # 처리 기관
date = cols[3].text.strip()      # 날짜

# 정보 출력
print("번호 : ", doc_num)
print("제목 : ", title)
print("처리 기관 : ", dept)
print("날짜 : ", date)

# 제목 클릭하여 상세 페이지로 이동
title_link = cols[1].find_element(By.TAG_NAME, "a")
title_link.click()

# 상세 페이지 로딩 대기
time.sleep(2)

# 본문 추출 (.b_cont 클래스를 가진 요소들)
contents = driver.find_elements(By.CSS_SELECTOR, ".b_cont")
full_text = "\n\n".join([c.text.strip() for c in contents])

# 본문 출력
print("본문 : ", full_text)

# 브라우저 종료
driver.quit()

