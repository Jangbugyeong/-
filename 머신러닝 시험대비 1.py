import numpy as np
import time
a = np.array([2,3,4])
print(a)

store_a = [20,10,30] # 매장 A의 매출 : 파이썬 리스트로 표현
store_b = [70,90,70] # 매장 B의 매출 : 파이썬 리스트로 표현
list_sum = store_a+store_b # 파이썬 리스트의 더하기 연산
print(list_sum)

np_store_a = np.array(store_a) # store_a 리스트를 넘파이 배열로 변환
np_store_b = np.array(store_b) # store_b 리스트를 넘파이 배열로 변환

array_sum = np_store_a+np_store_b
print(array_sum)

# a 객체의 형상(shape), 차원, 요소의 자료형, 요소의 크기(byte), 요소의 수
print(a.shape, a.ndim, a.dtype, a.itemsize, a.size)

b = np.array([[1, 2, 3], [4, 5, 6]]) # 넘파이 ndarray 객체의 b 생성
print(b.shape)

a = np.array([10, 20, 30])   # 넘파이 ndarray 객체 a 생성
b = np.array([1, 2, 3])      # 넘파이 ndarray 객체 b 생성

print(a+b) # 값 : [11 22 33]
print(a*b) # 값 : [10 40 90]

# 넘파이 배열의 데이터 타입을 지정하는 두 가지 방법
a = np.array([1,2,3,4], dtype = np.int32) # dtype = np.int32와 같이 np의 int32 속성값으로 지정하기
a = np.array([1,2,3,4], dtype = 'int32') # 문자열 형식으로 속성값 지정하기

# 3.4 다차원 배열과 브로드캐스팅
# 브로드캐스팅은 스칼라 n을 벡터로 확장시켜주는 작업이다.
a = np.array([10,20,30])
print(a*10) # 값 : [100 200 300]

b = np.array([[10,20,30],[40,50,60]])
c = np.array([2,3,4])
print(b+c) # 값 : [[12 23 34]
                  #[42 53 64]]
print(b*c) # 값 : [[ 20  60 120]
                  #[ 80 150 240]]

np.zeros((2,3)) # 2행 3열의 행렬 생성시 모든 값을 0으로
np.ones((2,3)) # 2행 3열의 행렬 생성시 모든 값을 1로
np.full((2,3),100) ## 2행 3열의 행렬 생성시 모든 값을 100으로
np.eye(3) # 3x3 크기의 단위 행렬 생성

# 3.5 연속적인 값을 가지는 다차원 배열의 생성
np.arange(0,10) # 0에서 9까지 연속적인 수열을 생성
np.arange(0,10,2) # 0에서 9까지 2씩 증가하는 수열을 생성
np.arange(0.0, 1.0, 0.2) # 0.2씩 증가하는 수열을 생성
np.linspace(0, 10, 5) # 0에서 10까지 5개로 나누어 일정한 값으로 증가하는 수열을 생성
np.linspace(0, 10, 4) # 0에서 10까지 4개로 나누어 일정한 값으로 증가하는 수열을 생성
a = np.logspace(0, 5, 6).round(10) # 10^0에서 10^5까지 6개의 수를
                                  # 로그 스케일 균일한 간격으로 생성한다.
np.log10(a) # a 배열에 로그를 취하면 0, 1, 2, 3, 4, 5가 생성됨

#3.6 다차원 배열의 축과 삽입
a = np.array([1,3,4])
np.insert(a,1,2) # a의 배열에 (0,1)의 자리에 2를 생성

b = np.array([[1, 1], [2, 2], [3, 3]])
np.insert(b,1,4,axis = 0) # b의 배열의 (1,0)의 자리에 4를 생성 가로로
np.insert(b, 1, 4, axis = 1) # b의 배열의 가운데에 각각 4를 생성

c = np.array([[1, 2, 3], [4, 5, 6]])
np.flip(c, axis=1) # c의 배열의 순서를 각각 뒤집음 [3,2,1],[6,5,4]
np.flip(c, axis=0) # 배열 c의 행의 순서를 뒤바꿈 [4,5,6],[1,2,3]

a = np.array( [[1, 2, 3], [4, 5, 6],
               [7, 8, 9], [0, 1, 2]] )
print(a)
print(a[0][0]) #첫 번째 리스트에 있는 첫 번째 값을 반환
print(a[2]) # 세 번째 리스트를 반환

print(a[1:][0:2]) # 첫 번째 리스트를 슬라이싱한 다음 첫 번째 리스트부터 두 번째 리스트까지 반환

#넘파이 스타일의 슬라이싱과 논리 인덱싱

a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [0, 1, 2]])
print(a[1:,0:2]) # 배열 a의 1,2,3을 슬라이싱하고 각 리스트의 세 번째 숫자를 슬라이싱

a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]])
print(a % 2 == 0) #a를 2로 나눈 나머지가 0이면 True를 반환, 아니면 False를 반환
print(a[ a % 2 == 0 ]) #를 2로 나눈 나머지가 0인 리스트 안의 원소를 반환

a = np.array([10, 20, 30])
a.max(), a.min(), a.mean() # a 배열원소 중 최대값, 최소값, 평균값을 출력함
a.astype(np.float64) # a안의 리스트 요소의 데이터 타입을 float64로 바꿈

b = np.array([[1, 1], [2, 2], [3, 3]])
b.flatten() # 넘파이 다차원 배열의 평탄화 메소드 [1,1,2,2,3,3] 2차원 배열을 1차원으로 바꿈
b.T # 배열의 가로와 세로를 바꿈 [1,2,3],[1,2,3]

c = np.array([35, 24, 55, 69, 19, 99])
c.sort() # 배열 정렬하는 메소드(기본:오름차순)
c[::-1] # 내림차순 정렬

d = np.array([[35, 24, 55], [69, 19, 9], [4, 1, 11]])
d.sort()         # 배열의 정렬, 디폴트 axis는 1

d = np.array([[35, 24, 55], [69, 19, 9], [4, 1, 11]])
d.sort(axis=0) # axis = 0 방향으로 정렬

# 3.10 다차원 배열을 위한 append()함수와 행렬의 곱셈

a = np.array([1, 2, 3])
b = np.array([[4, 5, 6], [7, 8, 9]])
np.append(a,b) # 1차원 배열로 바꾸면서 리스트를 더함 [1,2,3,4,5,6,7,8,9]
np.append([a],b,axis=0) # [a]를 통해 2차원 배열로 만들어야 함
a = [1, 2, 3, 4, 5, 6, 7, 8] # 1차원 리스트 a
b = np.reshape(a,(2,4)) # a를 2행 4열의 행렬 b로 바꿈
c = np.reshape(a,(4,2)) # 4행2열 행렬 c
a = np.arange(1,9)   # 1차원 배열 a
b = a.reshape(2,2,2) # 3차원 배열 b
a=np.arange(10).reshape(-1,5) # -1의 뜻은 리스트 원소의 개수에 따라 자동으로 맞춰주는 역할
a.reshape(-1) # [0,1,2,3,4,5,6,7,8,9]

# 3.11 난수

def pseudo_rand(x):
    a = 1103515245
    b = 12345
    m = 2 ** 31
    new_x = (a * x + b) % m
    return new_x

seed = 100 # 초기값을 임의로 설정하자

x = pseudo_rand(seed) # seed를 입력으로 하여 새로운 x를 만들자
print(x)
x = pseudo_rand(x) # x_n을 입력으로 하여 새로운 x_(n+1)을 만들자
print(x)

np.random.rand(5) # 0에서 1 사이의 난수 5개를 생성하는 넘파이 명령
np.random.randint(150, 191, size=10)
# (최소값,최대값,개수) 최대값이 191일 경우 191은 원소에 포함되지 않음
rnd = np.random.randn(5) * 10 +165 # 평균값이 165이고 표준편차가 10인 값 5개 생성
rnd.round(2) # 소수점 아래 둘째 자리까지 값으로 변환
rnd.astype(int)    # 정수형 자료형으로 변환
np.random.seed(42) # 시드 값이 특정될 경우 매번 동일한 난수의 열이 생성됨
nums = np.random.normal(loc=165, scale=10,size=(3,4)),round(2)
# 평균값 165, 표준편차 10,배열의 형태(3,4),소수점 둘째 자리까지 반환하는 함수
a = np.arange(10)
np.random.shuffle(a) # a의 값의 순서를 뒤섞어서 배치함
np.random.permutation([2,4,6,8,10]) # [2,4,6,8,10]의 배열을 랜덤으로 뒤섞음

# 평균,분산,표준편차

a = np.array([10, 20, 30, 40, 50])
a.mean() # 평균
A = np.array([60, 70, 80, 90, 100])
np.var(A) # A 모둠의 분산
B = np.array([76, 78, 80, 82, 84])
np.var(B) # B 모둠의 분산

#3.15 리덕션 : 배열을 더 강력하게 만드는 기능
arr = np.array([[1,2,3,4,5],
                [1,2,3,4,5],
                [1,2,3,4,5],
                [1,2,3,4,5],
                [1,2,3,4,5],
               ])
np.sum(arr,axis=0) # (0.n),(1,n)...의 방식으로 덧셈하는 함수 [5,10,15,20,25]
np.sum(arr,axis=1) # (n,0),(n,1)..의 방식으로 덧셈하는 함수 [15,15,15,15,15]
np.sum(arr) # 2차원 배열 전체를 더하는 함수 75

n = 1000
arr = np.random.rand(n,n)

#리스트 방식으로 함수를 써서 계산했을 때 걸리는 시간을 구한 함수
sum_naive = 0
start = time.time()
for i in range(n):
    for j in range(n):
        sum_naive += arr[i,j]
end = time.time()
print(n, 'x', n, "배열의 원소 합", sum_naive, "\n계산 시간: ", end-start)

#넘파이 방식으로 함수를 써서 계산했을 때 걸리는 시간을 구한 함수
start = time.time()
sum_reduction = np.sum(arr, axis = (0,1))
end = time.time()
print(n, 'x', n, "배열의 원소 합", sum_reduction,"\n계산 시간: ",end-start)

#3.16 배열의 결합 concatenate, vstack, hstack
a = np.arange(10, 18).reshape(2,4) # (2,4)형상을 가진 다차원 배열로 10에서 17까지
b = np.arange(12).reshape(3,4) # (3,4)형상을 가진 다차원 배열 0에서 11까지
c = np.arange(8).reshape(2,4)
d = list(range(4)) #d는 1차원 리스트로 0에서 3까지의 원소를 가짐
np.concatenate((a,b)) # a 배열의 마지막 원소 다음에 b 배열을 넣는다.
#array([[10, 11, 12, 13],
       #[14, 15, 16, 17],
       #[ 0,  1,  2,  3],
       #[ 4,  5,  6,  7],
       #[ 8,  9, 10, 11]])
np.vstack((a,b)) # a 배열과 b 배열을 수직 방향으로 결합,값은 전과 같음
np.vstack((a,d)) # a 베열과 d 리스트를 수직 방향으로 결합
#array([[10, 11, 12, 13],
       #[14, 15, 16, 17],
       #[ 0,  1,  2,  3]])
np.vstack((a, b, d)) # 다차원 배열 3개를 결합
np.hstack((a, c))   # 다차원 배열 a, c를 수평으로 결합
#array([[10, 11, 12, 13,  0,  1,  2,  3],
 #      [14, 15, 16, 17,  4,  5,  6,  7]])
# a 배열은 형상이 (2, 4), b 배열은 (3, 4)로 수평 결합이 안됨
np.hstack((a, b)) # a 배열과 b 배열의 hstack 결합-오류,형상이 안맞으면 배열의 결합 오류 발생