import numpy as np   # 반복되는 내용으로 앞으로 삭제함

np.random.seed(85)   # 동일한 결과를 얻기위해 85라는 초기값 사용
x = np.arange(0, 10) # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9를 생성
y1 = x * 2           # 0, 2, 4, 6, 8, 10, 12, 14, 16, 18
np.corrcoef(x, y1) #x와 y1과의 상관계수를 보여주는 함수. 1이면 양의 상관관계,-1이면 음의 상관관계,0이면 관계 없음.

x = np.arange(0, 10) # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9를 생성
y2 = x ** 3          # x의 세제곱 값을 원소로 함
np.corrcoef(x, y2) # x와 y2와 상관계수. [1,0]과 [0,1]이 상관계수임.

x = np.arange(0, 10)                     # 0에서 9사이의 연속적인 수를 생성
y3 = np.random.randint(0, 100, size=10)  # 0에서 1000사이 10개의 난수 생성
np.corrcoef(x, y3) #0 이상 100 미만의 정수 중에서 무작위로 10개를 생성해서, 배열 y3에 저장하는 코드

result = np.corrcoef((x, y2, y3))
print(result)

import matplotlib.pyplot as plt

plt.imshow(result)
plt.colorbar()
### 5.4 특성간의 관련성을 알려주는 상관계수와 쌍그래프

import seaborn as sns
import pandas as pd

df = pd.DataFrame( {'x': x, 'y2': y2,'y3': y3}) # x, y2, y3라는 세 가지 변수로 구성된 **표(데이터프레임)**를 만든다.
# #예: 열(column) 이름이 'x', 'y2', 'y3'이고 각 열에 해당하는 데이터가 들어감.
sns.pairplot(df) # pairplot()은 데이터프레임의 모든 변수 조합에 대해:산점도,히스토그램,관계 시각화를 자동으로 그려줌.
print('x =', x)
print('y2 =', y2)
print('y3 =', y3)

#5.5 시본 라이브러리 시작하기

sns.set_theme(style="darkgrid") # Seaborn의 **기본 테마(스타일)**를 설정하는 함수

tips = sns.load_dataset("tips") # Seaborn에서 제공하는 예제 데이터셋 중 하나인 "tips"를 불러오는 코드
sns.relplot(x="total_bill", y="tip", data=tips) # relplot은 관계(Relational) 시각화 함수,x="total_bill": x축은 전체 식사 금액

sns.relplot(x="total_bill", y="tip", hue="smoker", data=tips)
sns.relplot(x="total_bill", y="tip", hue="smoker", style="smoker",
            data=tips)

# 테마를 설정하는 기능이 있음
sns.set_theme(style="darkgrid")
# 시본에서 제공하는 팁 데이터를 가져오자
tips = sns.load_dataset("tips")

# 팁 데이터의 시각화 기능을 호출하자
sns.relplot(data=tips, x="total_bill", y="tip", col="time",
            hue="smoker", style="smoker", size="size")

# 5.6 tips 데이터와 여러가지 시각화 방법
# tips 데이터는 판다스 데이터프레임이다
tips.head(7) # tips의 맨 위 7칸의 데이터를 보이기
tips.shape # tips의 형태를 보이기

tips['tip_pct'] = 100 * tips['tip'] / tips['total_bill'] # tips의 퍼센트를 보이기 위해 tip_pct를 만듦
sns.histplot(tips['tip_pct'], kde=True, bins=20) # 구간을 20개로 나누어 히스토그램을 보이기
# kde=True: 히스토그램 위에 **밀도 곡선(Kernel Density Estimation)**을 그려서 분포 모양을 부드럽게 보여줌

# 5.7 산점도 그래프로 관계를 상세하게 나타내보자
sns.relplot(x='total_bill', y='tip', data=tips)

sns.relplot(x='total_bill', y='tip', hue='smoker', # style='smoker'	점의 모양도 흡연자 여부에 따라 다르게
            style='smoker', data=tips) # hue='smoker'	점의 색깔을 흡연자 여부(Yes or No)에 따라 다르게
sns.relplot(data=tips, x="total_bill", y="tip", col="time", # 시간별로 나눠서 그리기
            hue="smoker", style="smoker", size="size") # 횟수에 따른 사이즈 맞추기

ax = sns.regplot(data=tips, x='total_bill', y='tip') #x축과 y축 변수의 **관계(산점도)**를 시각화하면서,#
# **선형 회귀선(linear regression line)**을 자동으로 추가해줌
ax.set_xlabel('Total Bill')  # x 축의 레이블
ax.set_ylabel('Tip')         # y 축의 레이블
ax.set_title('Total Bill and Tip')   # 그림의 제목


# 5.10 비선형 함수를 사용하여 데이터를 설명하자

anscombe = sns.load_dataset("anscombe") # anscombe는 통계학에서 유명한 예제 데이터셋.4개의 데이터셋(‘I’, ‘II’, ‘III’, ‘IV’)이 있다.
sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'II'"), # 데이터셋 중 "dataset == 'II'" 조건을 만족하는 두 번째 그룹만 필터링
           order=2, ci=None, scatter_kws={"s": 80}) # 회귀선의 차수를 **2차(polynomial regression)**로 설정

sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'III'"), # lmplot()은 산점도 + 회귀선을 함께 그리는 함수
           ci=None, scatter_kws={"s": 80}) #회귀선 주변에 그려지는 **신뢰 구간(CI: confidence interval)**을 표시하지 않음
# 산점도의 **점 크기(size)**를 키움
sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'III'"),
           robust=True, ci=None, scatter_kws={"s": 80}) # robust=True는 회귀선이 이상치에 덜 흔들리도록 "강건한" 회귀선을 그려주는 옵션