#4.1 데이터 과학과 효과적인 시각화의 필요성
import random

# 데이터 시각화 : 컴퓨터 화면에 시각적 이미지를 이용하여 데이터를 효과적으로 보여주는 방법
# 사람들은 수치 데이터보다 시각적으로 보이는 그림을 더 직관적으로 이해할 수 있음

#4.2 데이터 과학을 위한 시각화 도구 matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
import holoviews.examples.gallery.demos.bokeh.lesmis_example
from scipy.stats import norm

plt.plot([1,2,3,4]) # 넣을 자료
plt.ylabel('y label') # y축 이름
plt.xlabel('x label') # x축 이름
plt.show() # 그래프 보이기

x = np.arange(10) # 0부터 9까지의 정수값 생성
plt.plot(x**2) # x제곱의 함수를 그림 0부터 9까지

x = np.arange(10)
plt.plot(x**2)
plt.axis([0,100,0,100])
plt.show()

# 4.3 plot() 함수의 선그리기 기능들을 알아보자

x = np.arange(-20,20) # -20에서 20까지의 수를 1의 간격으로 표현
y1 = 2 * x # 2x를 원소로 가지는 함수
y2 = (1/3) * x ** 2 + 5 # (1/3) * x 제곱 + 5의 함수 y2
y3 = -x ** 2 - 5 # -x ** 2 - 5 의 함수 y3
# 빨강색 점선, 녹색 실선과 세모기호, 파랑색 별표와 점선으로 함수를 표현
plt.plot(x,y1,'g--',x,y2, 'r^-',x,y3,'b*:')
plt.axis([-30,30,-30,30]) # 그림을 그릴 영역을 지정함

# 4.4 복잡한 선을 그리고 이미지로 저장하자

n=50
x=np.arange(n)
y = np.random.random(size=n)
plt.plot(x,y,'g^:') # 녹색의 점선,삼각형의 표식

x = np.linspace(0,np.pi * 2,100) # 0에서 2파이(360도)까지 100씩 띄움
plt.plot(x,np.sin(x),'r-') # 사인함수는 빨강 실선으로
plt.plot(x,np.cos(x),'b:') # 코사인함수는 파랑 점선으로
#위와 같은 과정이지만 피규어 함수로 저장함
x = np.linspace(0, np.pi * 2, 100)
fig = plt.figure() # fig 변수는 이 함수를 사진으로 저장하는 함수
plt.plot(x, np.sin(x), 'r-')
plt.plot(x, np.cos(x), 'b:')
fig.savefig('sin_cos_fig.png') # 이 이름으로 저장

Image('sin_cos_fig.png') # 위에 저장한 함수 보이기

# 4.5 제목과 레이블, 스타일에 대해 알아보자

x = np.linspace(0, np.pi * 2, 100)
plt.title('sin cos curve') # 보이는 그림의 이름을 sin cos curve로 지정
plt.plot(x, np.sin(x), 'r-',label='sincurve') # 사인함수의 이름을 sincurve로 지정
plt.plot(x, np.cos(x), 'b:',label='coscurve') # 코사인함수의 이름을 coscurve로 지정
plt.xlabel('x value') # x축의 이름을 x value로 지정
plt.ylabel('y value') # y축의 이름을 y value로 지정
label='sin curve'
label='cos curve'
plt.legend() # 위의 label 변수 두개를 순서대로 그림 왼쪽 아래에 설명으로 넣음

plt.style.use('seaborn-v0_8-whitegrid')   # 배경 스타일 적용
x = np.linspace(0, np.pi * 2, 100)
plt.title('Sin cos curve')
plt.plot(x, np.sin(x), 'r-', label='sin curve')
plt.plot(x, np.cos(x), 'b:', label='cos curve')
plt.xlabel('x value')
plt.ylabel('y value')
label='sin curve'
label ='cos curve'
plt.legend()
print(plt.style.available) # 쓸 수 있는 plt.style.use 안의 코드를 전부 보이기
plt.style.use('default') # 기본 스타일로 복귀

fig,ax=plt.subplot(2,2) # 2행 2열의 그림을 그리는 공간을 만든다
X = np.random.randn(100) # 표준정규분포를 가지는 데이터 X
Y = np.random.randn(100) # 표준정규분포를 가지는 데이터 Y
ax[0,0].scatter(X,Y) # 1행 1열의 그림을 xy의 산점도 그림으로 그림
X = np.arange(10)
Y = np.random.uniform(1,10,10) #균일분포값 생성
ax[0,1].bar(X,Y) # 1행 2열의 위치에 막대그래프 그림을 생성
X = np.linspace(0,10,100) # 0에서 10까지 균일한 차이의 100개의 수를 만듦
Y = np.cos(X) # X의 수를 코사인함수로 표현
ax[1,0].plot(X,Y) # 2행 1열에 실선으로 함수를 그림
Z = np.random.uniform(0, 1, (5, 5))
ax[1,1].imshow(Z) # Z의 분포를 2D이미지로 그림

fig, ax=plt.subplot(2,3)
grid = plt.GridSpec(2,3,wspace=0.4,hspace=0.3)
X = np.random.randn(100) # 표준정규분포를 가지는 데이터 X
Y = np.random.randn(100) # 표준정규분포를 가지는 데이터 Y
plt.subplot(grid[0,0]).scatter(X,Y) #산점도 그림
X = np.arange(10) # 0에서 9 사이의 연속값
Y = np.random.uniform(1, 10, 10) # 균일분포값 생성
plt.subplot(grid[0,1:]).bar(X,Y) # 1행의 2열부터 마지막 열까지 막대 차트 그림 생성
X = np.linspace(0, 10, 100)
Y = np.cos(X)
plt.subplot(grid[1, :2]).plot(X, Y) # 2행의 첫번째 열부터 두번째 열까지 실선으로 함수를 그림
Z = np.random.uniform(0, 1, (5, 5)) # 0부터 1사이의 실수를 랜덤으로 뽑아서 5행 5열짜리 배열을 만듦
plt.subplot(grid[1,2]).imshow(Z) # 2행 3열에 Z의 분포를 2D 이미지로 그림

# 4.8 자료값의 분포를 나타내는 산점도와 막대 그래프

n = 30 # 전체 점들의 개수
x = np.random.rand(n) # 점들의 임의의 x 좌표
y = np.random.rand(n) # 점들의 임의의 y 좌표
colors = np.random.rand(n) # 랜덤한 색상
area = (30*np.random.rand(n))**2 # 지름이 0에서 30 포인트 범위
# 산포도 그림을 x,y 를 따라서 지름이 0부터 30인 colors의 명령을 따른 색깔로 오각형의 범위로 표현함,투명도 50
plt.scatter(x,y,s=area,c=colors,alpha=0.5,marker='p')

x = np.arange(3)
years = ['2010', '2011', '2012']
domestic = [6801, 7695, 8010]
foreign = [777, 1046, 1681]

plt.bar(x, domestic)   # 2010, 2011, 2012년의 domestic 값으로 바 차트 그리기
plt.bar(x, foreign)    # 2010, 2011, 2012년의 foreign 값으로 또 다른 바 차트 그리기
plt.xticks(x, years)   # x축에 2010, 2011, 2012년 표시하기

plt.bar(x, domestic, width=0.25)        # domestic 데이터를 바 차트로 그리되, 바의 너비는 0.25
plt.bar(x + 0.3, foreign, width=0.25)   # foreign 데이터를 바 차트로 그리되, 바의 위치는 x + 0.3만큼 이동
plt.xticks(x, years)                    # x축에 2010, 2011, 2012를 표시

x = np.arange(3)
years = ['2010', '2011', '2012']
domestic = [6801, 7695, 8010]
foreign = [777, 1046, 1681]

plt.barh(x, foreign, height=0.25)      # foreign 데이터를 수평 바 차트로 그리되, 바의 높이는 0.25
plt.barh(x + 0.3, domestic, height=0.25)  # domestic 데이터를 수평 바 차트로 그리되, 바의 위치는 x + 0.3만큼 이동
plt.yticks(x, years)                   # y축에 2010, 2011, 2012를 표시

# 앞의 함수에 foreign 값과 domestic의 값을 더한 sum 리스트를 만들어 그 값을 보이기

sum = [7578,8741,9691]

plt.bar(x, domestic, width=0.25)
plt.bar(x + .3, foreign, width= 0.25)
plt.bar(x +.6, sum, width = 0.25 )
plt.xticks(x,years)

# 4.9 파이 차트와 히트맵 표현

data = [5, 4, 6, 11] # 숫자열 리스트, 각 조각에 해당하는 값을 나타냄
clist = ['cyan', 'gray','orange', 'red'] # 문자열 리스트, 각 조각의 색깔을 지정함
plt.pie(data,autopct='%.2f',colors=clist) # 파이 차트를 그리는 함수,autopct: 각 조각의 비율을 소수점 2자리까지 표시.

data = [5, 4, 6, 11]
clist = ['cyan', 'gray','orange', 'red']
explode = [.06, .07, .08, .09]
# explode: 각 파이 조각을 얼마나 부풀릴지 설정하는 값. autopct: 각 조각의 비율을 소수점 두 자리까지 표시.
# labels: 각 파이 조각에 라벨을 붙이기 위한 값.
plt.pie(data,autopct='%.2f%%',colors=clist,labels=clist,explode=explode)

data = random.random((10,10)) # 임의의 수를 가진 10x10 크기의 배열
plt.imshow(data) # data의 분포를 2D 이미지로 그림
plt.colorbar() # 컬러바를 이용해 값을 크기를 보여줌

# 4.10 히스토그램

heights = np.array([175, 165, 164, 164, 171, 165, 160, 169, 164, 159, 163, 167, 163,172, 159, 160, 156, 162, 166, 162, 158, 167, 160, 161, 156, 172, 168, 165, 165, 177])
plt.hist(heights,bins=6) # heights의 리스트의 안에 있는 값들을 6개의 값의 구간으로 나눔
plt.xlabel("height") # x축의 이름으로 키를 나타냄
plt.ylabel("frequency") # y축의 이름으로 각 구간에 얼마나 많은 값이 분포해 있는지 표현

plt.hist(heights, bins=6, label='cumulative=True', cumulative=True) # 누적 히스토그램으로 표현하겠다는 옵션, 누적 히스토그램은 각 구간의 빈도수가 그 이전 구간들을 전부 더한 값으로 나타냄
plt.hist(heights, bins=6, label='cumulative=False', cumulative=False) # 일반 히스토그램으로 표현한다는 옵션, 각 구간의 빈도수만 나타내는 일반적인 히스토그램
plt.legend(loc='upper left') # 설명을 붙이는 위치를 기본값이 아닌 왼쪽 위로 지정함

# 4.11 히스토그램을 이용한 정규 분포 함수와 확률 밀도 함수 그리기

f1 = np.random.randn(100000) # 정규분포를 따르는 랜덤한 숫자 십만 개 생성
plt.hist(f1,bins=200,color='red',alpha=.7,label='avg=0, std=1')
plt.axis([-8,8,-2,2500])
plt.legend()

f1 = np.random.normal(loc=0, scale=1, size=100000) # 평균이 0 이고 표준편차가 1인 장규 분포에서 랜덤하게 100000개의 데이터를 생성함
f2 = np.random.normal(loc=3, scale=1, size=100000) # 평균이 3이고 표준편차가 1인 정규 분포에서 랜덤하게 10000개의 데이터를 생성함

plt.hist(f1, bins=200, color='red', alpha=.4,
         label='avg = 0, std = 1') # f1의 데이터에서 구간을 200개로 나누어 빨간색으로 투명도 40인 히스토그램을 그림
plt.hist(f2, bins=200, color='green', alpha=.4,
         label='avg = 3, std = 1') # f2의 데이터에서 구간을 200개로 나누어 초록색으로 투명도 40의 히스토그램을 그림
plt.axis([-8, 8, -2, 2500]) # x 축과 y축의 범위를 설정하는 함수 x축 범위는 -8에서 8까지,y축 범위는 -2에서 2500까지
plt.legend() # 범례를 화면에 표시하는 함수

f1 = np.random.normal(loc=0, scale=1, size=100000) # 평균이 0이고 표준편차가 1인 정규분포 데이터 100000개 생성
f2 = np.random.normal(loc=3, scale=3, size=100000) # 평균이 3이고 표준편차가 3인 정규분포 데이터 100000개 생성

plt.hist(f1, bins=200, color='red', alpha=.4,
         label='avg = 0, std = 1') # f1의 데이터에서 구간을 200개로 나누어 빨간색으로 투명도 40인 히스토그램을 그림
plt.hist(f2, bins=200, color='green', alpha=.4,
         label='avg = 3, std = 3') # f2의 데이터에서 구간을 200개로 나누어 초록색으로 투명도 40의 히스토그램을 그림
plt.axis([-8, 8, -2, 2500]) # x 축과 y축의 범위를 설정하는 함수 x축 범위는 -8에서 8까지,y축 범위는 -2에서 2500까지
plt.legend() # 범례를 화면에 표시하는 함수

f1 = np.random.normal(loc=0, scale=1, size=10000) # 평균이 0이고 표준편차가 1인 랜덤한 데이터 10000개 생성
f2 = np.random.normal(loc=3, scale=.5, size=10000) # 평균이 3이고 표준편차가 5인 랜덤한 데이터 10000개 생성

plt.hist(f1, bins=200, color='red', alpha=.7,
         label='loc = 0, scale = 1') #f1의 데이터에서 구간을 200개로 나누어 빨간색으로 투명도 70의 데이터를 그림
plt.hist(f2, bins=200, color='blue', alpha=.5,
         label='loc = 3, scale = .5') # f2의 데이터에서 구간을 200개로 나누어 파란색으로 투명도 50인 그림을 그림


data = norm.rvs(10.0,3,size=1000) # 평균이 10.0이고 분산이 3인 1000개의 데이터를 생성
plt.hist(data, bins=20,density=True,alpha=0.6,color='b') # data의 데이터로 구간을 20개로 나누어 투명도 60퍼센트, 파란색으로 칠함
mu,std = norm.fit(data) # 정규 분포를 데이터에 피팅함
xmin, xmax = plt.xlim()  # x의 최소, 최대값을 pyplot으로 부터구함
x = np.linspace(xmin, xmax, 100) # x의 최소값부터 최대값까지 균등한 간격으로 100개의 숫자를 생성
p = norm.pdf(x,mu,std) # scipy의 pdf는 평균, 표준편차로부터 확률 밀도 함수를 생성한다
plt.plot(x, p, 'r', linewidth=2) # 빨간색으로 그리고 라인의 너비를 2로 설정함

#4.13 상자 수염 그리기

rand_data = np.random.randn(100) # 넘파이의 난수 모듈에 있는 정규 분포에 따르는 난수 100개를 생성
plt.boxplot(rand_data) # 박스 모양으로 그림. 주황색 선은 중앙값, 위의 선은 상위 25퍼센트의 값, 아래의 선은 하위 25퍼센트의 값, 맨 위의 원과 아래의 원이 있다면 이상치가 있다는 뜻, 위아래로 길어진 선의 길이는 분산의 크기를 뜻함

np.random.seed(85)
data1 = np.random.normal(100, 10, 200) # 평균이 100이고 분산이 10
data2 = np.random.normal(100, 40, 200) # 평균이 100이고 분산이 40
data3 = np.random.normal(80, 40, 200)  # 평균이 80이고 분산이 40
data4 = np.random.normal(80, 60, 200)  # 평균이 80이고 분산이 60

plt.boxplot([data1, data2, data3, data4])

#4.14 그래프의 크기와 그리드 그리기

X = np.linspace(0,2*np.pi,200)
Y = np.sin(x)
plt.figure(figsize=(4.2,3.6)) # 그래프의 크기는 4.2인치 x 3.6인치로 설정
plt.plot(X,Y) # 이 데이터를 기반으로 사인 함수 그래프가 그려짐
plt.grid() # 배경에 격자 선이 추가됨
# plt.grid() 대신 다음과 같은 키워드 인자를 지정하여 호출
X = np.linspace(0, 2*np.pi, 200)
Y = np.sin(X)
plt.figure(figsize=(4.2, 3.6))
plt.plot(X, Y)
plt.grid(color='r', linestyle='dotted', linewidth=2) # 빨간색의 점선으로 선의 너비는 2로 해서 격자를 생성함

plt.grid(axis = 'y',color = 'r', linestyle = 'dotted',linewidth = 2) # 빨간색의 점선으로 y축으로만 구분한 선을 2의 너비로 생성함
plt.grid(axis = 'x',color = 'r', linestyle = 'dotted',linewidth = 2) # 빨간색의 점선으로 x축으로만 구분한 선을 2의 너비로 생성함

















