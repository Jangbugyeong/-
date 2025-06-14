import numpy as np
from sklearn import linear_model  # scikit-learn 모듈을 가져온다

regr = linear_model.LinearRegression()

X = [[163], [179], [166], [169], [170]]
y = [54, 63, 57, 56, 58]
regr.fit(X, y)

coef = regr.coef_            # 직선의 기울기
intercept = regr.intercept_  # 직선의 절편
score = regr.score(X, y)     # 학습된 직선이 데이터를 얼마나 잘 따르나

print(f"y = {coef.round(2)}* X + {intercept:.2f}")
print(f"데이터와 선형회귀 직선의 관계점수: {score:.1%}")

# 7.7 데이터를 시각화하고 차원을 증가시키자

import matplotlib.pyplot as plt
# 학습 데이터와 y 값을 산포도로 그린다.
plt.scatter(X, y, color='blue', marker='D')
# 학습 데이터를 입력으로 하여 예측값을 계산한다.
y_pred = regr.predict(X)
plt.plot(X, y_pred, 'r:') # 예측값을 이어주는 점선을 그려보자

unseen = [[167]]
result = regr.predict(unseen)
print(f'동윤이의 키가 {unseen}cm 이므로 몸무게는 {result.round(1)}kg으로 추정됨')

from sklearn.metrics import r2_score
print(f"데이터와 선형 회귀 직선의 r square 점수: {r2_score(y, y_pred):.3}")

from sklearn import linear_model

regr = linear_model.LinearRegression()
# 남자는 0, 여자는 1
X = [[168, 0], [166, 0], [173, 0], [165, 0], [177, 0], [163, 0], \
     [178, 0], [172, 0], [163, 1], [162, 1], [171, 1], [162, 1], \
     [164, 1], [162, 1], [158, 1], [173, 1], ]    # 입력데이터를 2차원으로 만들어야 함
# y 값은 1차원 데이터
y = [65, 61, 68, 63, 68, 61, 76, 67, 55, 51, 59, 53, 61, 56, 44, 57]
regr.fit(X, y)         # 학습
print('계수 :', regr.coef_ )
print('절편 :', regr.intercept_)
print('점수 :', regr.score(X, y))
print('동윤이와 은지의 추정 몸무게 :', regr.predict([[167, 0], [167, 1]]))

# 위의 결과들을 3차원 좌표공간에 그리기


import matplotlib.pyplot as plt
import numpy as np


X = [[168, 0], [166, 0], [173, 0], [165, 0], [177, 0], [163, 0], \
     [178, 0], [172, 0], [163, 1], [162, 1], [171, 1], [162, 1], \
     [164, 1], [162, 1], [158, 1], [173, 1] ]    # 입력데이터를 2차원으로 만들어야 함
y = [65, 61, 68, 63, 68, 61, 76, 67, 55, 51, 59, 53, 61, 56, 44, 57]

X = np.array(X)

xs = X[:,0]
print(xs)
ys = X[:,1]
print(ys)
zs = np.array(y)
print(zs)


# Plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(xs, ys, zs)
ax.scatter([167,167],[0,1],[63.6,56.4], color = 'red')
ax.set_xlim(160, 180)
ax.set_ylim(0, 1)
ax.set_zlim(50, 80)

ax.set_xlabel('xs')
ax.set_ylabel('ys')
ax.set_zlabel('zs')

plt.show()

