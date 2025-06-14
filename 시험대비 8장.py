#8.7 데이터의 분포가 직선이 아니라면? 다항 회귀 모델을 사용하자

import numpy as np
import matplotlib.pyplot as plt

m = 100    # 생성할 데이터의 개수
# 평균값이 0이고 -4에서 4사이에 분포하는 랜덤 값 X
np.random.seed(84)   # 매번 동일한 결과를 얻기 위하여 시드를 84로 부여하자
X = 8 * np.random.rand(m, 1) - 4
# x^2항의 계수가 0.5, x항의 계수가 2, 상수항의 계수가 1
y = 0.5 * X ** 2 + 2 * X + 1 + np.random.randn(m, 1)

plt.figure(figsize=(6,4))
plt.plot(X, y, "b^")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

regr = LinearRegression()
regr.fit(X, y)
print('선형회귀 모델의 점수 =', regr.score(X, y))
print('선형회귀 모델의 계수와 절편 =', regr.coef_, regr.intercept_)

plt.figure(figsize=(7,6))
y_predict = regr.predict(X)
plt.scatter(X, y)
plt.plot(X.flatten(), y_predict, color='r')

# 8.8 다항 회귀 모델을 사용하자

import numpy as np
from sklearn.preprocessing import PolynomialFeatures



s = [2]   # 간단한 샘플 데이터 s를 생성(1차원)
print(s)

t = np.arange(6).reshape(3, -1)   # 간단한 샘플 데이터 t를 생성(2차원)
print(t)

poly = PolynomialFeatures(degree=2)       # 디폴트 degree=2임

new_s = poly.fit_transform([s])
print(new_s)

print(poly.get_feature_names_out())

poly = PolynomialFeatures(degree=2)       # 디폴트 degree=2임
new_t = poly.fit_transform(t)
print(new_t)        # t에 대하여 다항 특성을 추가하고 이를 출력해 보자
print(poly.get_feature_names_out())

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 8.7 절의 데이터 X
np.random.seed(42)
m = 100    # 생성할 데이터의 갯수
# 평균값이 0이고 -4에서 4사이에 분포하는 랜덤 값 X
X = 8 * np.random.rand(m, 1) - 4
# x^2항의 계수가 0.5, x항의 계수가 2, 상수항의 계수가 1
y = 0.5 * X ** 2 + 2 * X + 1 + np.random.randn(m, 1)

poly_features = PolynomialFeatures(degree = 2,\
                                   include_bias = False)
X_poly = poly_features.fit_transform(X)
regr = LinearRegression()
regr.fit(X_poly, y)
print('다항 회귀 모델의 점수 =', regr.score(X_poly, y))
print('다항 회귀 모델의 계수 =', regr.coef_, '절편 =', regr.intercept_)

# -4에서 4사이의 데이터를 생성하자.
domain = np.linspace(-4, 4, 50).reshape(-1, 1)
# domain 데이터에 2차 다항 특성을 추가한 domain_2를 만들자
domain_2 = poly_features.fit_transform(domain)
print(domain.shape, domain_2.shape)
print(domain[:3])
print(domain_2[:3])
print(poly_features.get_feature_names_out())
plt.figure(figsize=(6,4))
y_predict = regr.predict(domain_2)
plt.scatter(X, y)
plt.plot(domain, y_predict, color='r', linewidth=4)

# 2개의 변곡점을 가진 데이터를 생성하고 시각화하는 코드
import numpy as np
import matplotlib.pyplot as plt

m = 100
X = 8 * np.random.rand(m, 1) - 4
y = .5 * X ** 3 + .5 * X ** 2 + X + 3 + np.random.randn(m, 1)
plt.figure(figsize=(6,5))
plt.plot(X, y, "b^")

# 다항 회귀 모델을 만들고 점수와 계수, 절편을 출력하는 코드
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_features = PolynomialFeatures(degree = 3,\
                                   include_bias = False)
X_poly = poly_features.fit_transform(X)
regr = LinearRegression()
regr.fit(X_poly, y)

print('다항 회귀 모델의 점수 =', regr.score(X_poly, y))
print('다항 회귀 모델의 계수 =', regr.coef_)
print('절편 =', regr.intercept_)

# 다항 회귀 곡선을 생성하는 기능
X_new = np.linspace(-4, 4, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = regr.predict(X_new_poly)

plt.figure(figsize=(6,6))
plt.plot(X, y, "b^", label='Data')
plt.plot(X_new, y_new, "r-", label="Prediction")
plt.legend()