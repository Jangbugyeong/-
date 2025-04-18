import numpy as np
import random
a = np.arange(1,46)
print(a)
np.random.seed(85)
shuffled_a = np.random.permutation(a)
print(shuffled_a)
lotto = shuffled_a[:6]
print(lotto)
lotto.sort()
print(lotto)
print(lotto,f'추가번호 : {shuffled_a[6]}')
arr=np.arange(25)
n_arr=np.reshape(arr,(5,5))
print(n_arr)
print(f'첫 원소 : {n_arr[0][0]} 마지막 원소 : {n_arr[-1][-1]}')

print(n_arr[:2])
print(n_arr[: , [0,2,4]])
reshape_n_arr = n_arr[:2]
reshaped_n_arr=np.reshape(reshape_n_arr,(5,2))#(세로,가로) 순서
print(reshaped_n_arr)

a = np.random.rand(8)
reshape_a=np.reshape(a,(2,2,2))
print(reshape_a)
print('최대값 :',reshape_a.max())
print('최소값 : ',reshape_a.min())
a=np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])
print(a[:,1:]) #리스트 전부포함,각 리스트의 2번째 원소부터 전부 출력
print(a[0:1,0:2]) # 첫 번째 리스트에서 첫번째부터 두번째 원소까지 출력
print(a[::2,::2]) # [1,3,5][11,13,15]
b=np.arange(1,26)
reshape_b = np.reshape(b,(5,5))
print(reshape_b)
print(np.sum(reshape_b,axis=0)) # 행렬의 행 방향 성분의 합
print(np.sum(reshape_b,axis=1)) # 행렬의 열 방향 성분의 합

c = np.arange(32)
reshape_c=np.reshape(c,(4,4,2))
print(reshape_c)

print(f'10번째 숫자 : {reshape_c.flatten()[9]} 20번째 원소 : {reshape_c.flatten()[19]}')

