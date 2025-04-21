#6.2 시리즈의 자료형과 결손값

import numpy as np
import pandas as pd

se = pd.Series([1, 2, np.nan, 4])   # np.nan은 결측값
se.isna()    # np.nan이 포함된 세번째 항목만 True를 반환
se[0], se[1] # 리스트처럼 인덱싱 가능

data = [1, 2, np.nan, 4]
indexed_se = pd.Series(data, index = ['a', 'b', 'c', 'd'])
print(indexed_se)

indexed_se['a'], indexed_se['b']  # 인덱스가 'a', 'b',.. 임

income = {'1월' : 9500, '2월': 6200, '3월': 6050, '4월': 7000} # 딕셔너리
income_se = pd.Series(income)   #월을 인덱스로 하는 매출 시리즈
print('동윤이네 상점의 수익')
print(income_se)

month_se = pd.Series(['1월', '2월', '3월', '4월'])
income_se = pd.Series([9500, 6200, 6050, 7000])
expenses_se = pd.Series([5040, 2350, 2300, 4800])
df = pd.DataFrame( {'월': month_se, '수익': income_se,
                   '지출' : expenses_se})

print(df)

# 판다스 Series를 이용하여 최대 수익 월을 출력하기
m_idx = np.argmax(income_se)  # 넘파이의 argmax() 사용
print('최대 수익이 발생한 월:', month_se[m_idx])

print('월 최대 수익:', income_se.max(),
      ', 월 평균 수익:', income_se.mean())

path = 'https://github.com/dongupak/DataML/raw/main/csv/'
file = path+'vehicle_prod.csv'
df = pd.read_csv(file)   # 원격지에 접속하여 csv를 읽어옴

print(df)

df = pd.read_csv(file, index_col = 0)
print(df)

df.columns   # 데이터프레임의 컬럼값들을 살펴보자
df.index     # 데이터프레임의 인덱스값들을 살펴보자
df['2007']   # 데이터프레임의 2007년도 컬럼값들을 살펴보자
df.columns.tolist() # 컬럼값들을 리스트 형식으로 보이기

#6.6 새로운 열 생성

path = 'https://github.com/dongupak/DataML/raw/main/csv/'
file = path+'vehicle_prod.csv'
df = pd.read_csv(file, index_col=0)   # 원격지에 접속하여 csv를 읽어옴

df['total'] = df.sum(axis = 1) #total 이라는 이름을 가진 새로운 열을 생성
print(df)

# 2008년부터 2011년도 까지의 총 생산 대수와 평균을 다시 계산함
df['total'] = df[['2007', '2008', '2009', '2010', '2011']].sum(axis=1)
df['mean'] = df[['2007', '2008', '2009', '2010', '2011']].mean(axis=1)
print(df) # 수정 전 데이터를 계산한 값이 나오니 필수적으로 재계산을 해주어야 한다.

df.drop('2007', inplace=True, axis=1)
print(df)       # 이전에 구한 total, mean 값이 나타남

df.drop('2007', inplace=True, axis=1)
print(df)       # 이전에 구한 total, mean 값이 나타남

# 2008년부터 2011년도 까지의 총 생산대수와 평균을 다시 계산함
df['total'] = df[['2008', '2009', '2010', '2011']].sum(axis=1)
df['mean'] = df[['2008', '2009', '2010', '2011']].mean(axis=1)
print(df)       # 새로 계산한 total, mean 값이 나타남

### 6.7 inplace로 데이터프레임 갱신하기

d_df = pd.DataFrame(data = [[10, 20, 30, 40], [50, 60, 70, 80]],
                 columns = ['A', 'B', 'C', 'D'])
new_df = d_df.drop('B', axis=1, inplace=False) # drop 문을 써서 삭제시킴
print(new_df)    # 'B'열이 삭제된 새로운 데이터프레임
print(d_df)         # d_df 데이터프레임은 변화가 없음

d_df.drop('B', axis=1, inplace=True) # inplace 키워드 인자가 True라면
print(d_df)   # d_df 데이터프레임 내부의 값이 변경됨

path = 'https://github.com/dongupak/DataML/raw/main/csv/'
file = path+'vehicle_prod.csv'
df = pd.read_csv(file, index_col=0)   # 원격지에 접속하여 csv를 읽어옴

df.drop('Mexico', axis=0, inplace=True) # Mexico행을 삭제하고 df를 갱신
print(df)

# 6.8 데이터프레임 시각화

path = 'https://github.com/dongupak/DataML/raw/main/csv/'
file = path+'vehicle_prod.csv'

df = pd.read_csv(file, index_col = 0)
df_no_index = pd.read_csv(file)
print(df['2007'])
print(df_no_index['2007'])

print(df[['2007', '2008', '2009']]) # 2007년부터 2009년까지의 데이터만 프린트

df['2009'].plot(kind='bar', color=('orange','r', 'b', 'm', 'c', 'k')) # kind는 그래프의 형태

#6.9 편리하고 강력한 시각화

path = 'https://github.com/dongupak/DataML/raw/main/csv/'
file = path+'vehicle_prod.csv'
df = pd.read_csv(file, index_col=0)   # 원격지에 접속하여 csv를 읽어옴
df.plot.line() #df의 데이터프레임을 선 그래프로 시각화
df.plot.bar() # 수직 막대 그래프
df = df.transpose()  # df = df.T도 가능함 행과 열을 뒤바꿈
#df = df.T
print(df)
df.plot.line() # 행과 열을 바꾼 데이터프레임을 시각화

#6.10 편리한 데이터 다루기 - 슬라이싱과 인덱싱

df.head(3)     # 첫 3행의 정보를 가져온다

df[2:6] # 3행부터 6행까지의 정보를 가져온다

df.loc['Korea'] # 이름이 korea인 정보를 가져온다

df.loc[['US', 'Korea']]  # 주의: df.loc['US','Korea']는 오류

df['2011'][[0, 4]] # 2011년의 1행과 5행 가져오기

df.iloc[4]  # df.loc['Korea'] 와 동일함, df.iloc['Korea']는 오류,리스트 위치만 입력 가능

df.loc[['US','Korea']]  # 주의 :df.loc['US','Korea']는 오류

df.iloc[[2, 4]]  # 주의: df.iloc[2, 4]는 US행의 2011년 데이터임

df.iloc[2, 4]  # US행의 2011년 데이터를 출력

df.iloc[2:4]     # US행, Japan행의 데이터를 출력
df.iloc[[2, 3]]    # US행, Japan행의 데이터를 출력
df[2:4]             # US행, Japan행의 데이터를 출력

path = 'https://github.com/dongupak/DataML/raw/main/csv/'
weather_file = path + 'weather.csv'

weather = pd.read_csv(weather_file, index_col = 0, encoding='CP949')
print(weather.head(3))
print('weather 데이터의 shape :', weather.shape)

print(weather.describe())
print('평균 분석 -----------------------------')
print(weather.mean())
print('표준편차  분석 -----------------------------')
print(weather.std())
weather.count()

missing_data = weather[ weather['평균풍속'].isna() ]
print(missing_data)

# 결손값을 0으로 채움, inplace를 True로 설정해 원본 데이터를 수정
weather.fillna(0, inplace = True)
print(weather.loc['2012-02-11'])

weather.fillna( weather['평균풍속'].mean(), inplace = True)
print(weather.loc['2012-02-11'])

import matplotlib.pyplot as plt


