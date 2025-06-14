# 9.5 로지스틱 회귀 모델을 위한 데이터 만들기

import matplotlib.pyplot as plt

# 닥스훈트의 길이와 높이 데이터
dach_length = [55, 57, 64, 63, 58, 49, 54, 61]
dach_height = [30, 31, 36, 30, 33, 25, 37, 34]
# 진돗개의 길이와 높이 데이터
jin_length = [56, 47, 56, 46, 49, 53, 52, 48]
jin_height = [52, 52, 50, 53, 50, 53, 49, 54]

newdata_length = [59]    # 새로운 데이터의 길이
newdata_height = [35]    # 새로운 데이터의 높이

plt.scatter(dach_length, dach_height, c='r', label='Dachshund')
plt.scatter(jin_length, jin_height,c='b',marker='^', label='Jindo dog')
# 새 데이터의 표식은 오각형(pentagon)으로 설정하고, 레이블은 new Data로
plt.scatter(newdata_length, newdata_height, s=100, marker='p',\
           c='g', label='new Data')

plt.xlabel('Length')
plt.ylabel('Height')
plt.title("Dog size")
plt.legend(loc='upper right')

import numpy as np

d_data = np.column_stack((dach_length, dach_height))
d_label = np.zeros(len(d_data))   # 닥스훈트는 0으로 레이블링

j_data = np.column_stack((jin_length, jin_height))
j_label = np.ones(len(j_data))   # 진돗개는 1로 레이블링

# 9.6 로지스틱 회귀 모델 구현하기

import numpy as np

newdata = [[59, 35]]
dogs = np.concatenate((d_data, j_data))
labels = np.concatenate((d_label, j_label))
dog_classes = {0:'닥스훈트', 1:'진돗개'}

dogs[:5],labels[:5], dogs[-5:], labels[-5:]

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

f = LogisticRegression()
f.fit(dogs, labels)
w = f.coef_[0] # 계수
b = f.intercept_ # 절편
print('계수:', w)
print('절편:', b)
print('분류 정확도:', f.score(dogs, labels))

y_pred = f.predict(newdata) #predict 함수는 배열 형태로 리턴
print(y_pred)
y_pred_prob = f.predict_proba(newdata) #predict_proba 함수는 배열 형태로 리턴
print(f'데이터: {newdata}')
print(f'판정 결과: {dog_classes[y_pred[0]]}, 판정 확률: {y_pred_prob}')

plt.scatter(dach_length, dach_height, c='r', label='Dachshund')
plt.scatter(jin_length, jin_height,c='b',marker='^', label='Jindo dog')
# 새 데이터의 표식은 오각형(pentagon)으로 설정하고, 레이블은 new Data로
plt.scatter(newdata_length, newdata_height, s=100, marker='p',\
           c='g', label='new Data')

plt.xlabel('Length')
plt.ylabel('Height')
plt.title("Dog size")
plt.legend(loc='upper right')

print(np.min([dach_length, jin_length]))
print(np.max([dach_length, jin_length]))
x0 = np.linspace(np.min([dach_length, jin_length]),\
                 np.max([dach_length, jin_length]), 200)
decision_boundary = -w[0]/w[1] * x0 - b/w[1]
# 결정 경계에서 w0*x0 + w1*x1 + b = 0 이므로 => x1 = -w0/w1 * x0 - b/w1
plt.plot(x0, decision_boundary, "k-", linewidth=2)

dach_0 = w[0]*dach_length[0] + w[1]*dach_height[0] + b
jin_0 = w[0]*jin_length[0] + w[1]*jin_height[0] + b
print('닥스훈트 샘플 대입 결과:', dach_0)
print('진돗개 샘플 대입 결과:', jin_0)

# 9.7 가까운 이웃으로 분류하는 k-NN 분류기

import matplotlib.pyplot as plt

# 닥스훈트의 길이와 높이 데이터
dach_length = [55, 57, 64, 63, 58, 49, 54, 61]
dach_height = [30, 31, 36, 30, 33, 25, 37, 34]
# 진돗개의 길이와 높이 데이터
jin_length = [56, 47, 56, 46, 49, 53, 52, 48]
jin_height = [52, 52, 50, 53, 50, 53, 49, 54]

newdata_length = [59]    # 새로운 데이터의 길이
newdata_height = [35]    # 새로운 데이터의 높이

plt.scatter(dach_length, dach_height, c='r', label='Dachshund')
plt.scatter(jin_length, jin_height,c='b',marker='^', label='Jindo dog')

plt.xlabel('Length')
plt.ylabel('Height')
plt.title("Dog size")
plt.legend(loc='upper right')

# 새 데이터의 표식은 오각형(pentagon)으로 설정하고, 레이블은 new Data로
plt.scatter(newdata_length, newdata_height, s=100, marker='p',\
            c='g', label='new Data')

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# 닥스훈트의 길이와 높이 데이터
dach_length = [55, 57, 64, 63, 58, 49, 54, 61]
dach_height = [30, 31, 36, 30, 33, 25, 37, 34]
# 진돗개의 길이와 높이 데이터
jin_length = [56, 47, 56, 46, 49, 53, 52, 48]
jin_height = [52, 52, 50, 53, 50, 53, 49, 54]
newdata = [[59, 35]]

d_data = np.column_stack((dach_length, dach_height))
d_label = np.zeros(len(d_data))   # 닥스훈트는 0으로 레이블링
j_data = np.column_stack((jin_length, jin_height))
j_label = np.ones(len(j_data))   # 진돗개는 1로 레이블링

dogs = np.concatenate((d_data, j_data))
labels = np.concatenate((d_label, j_label))

dog_classes = {0:'닥스훈트', 1:'진돗개'}

k = 3     # k를 3으로 두고 kNN 분류기를 만들어 보자
knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(dogs, labels)
y_pred = knn.predict(newdata)
print(y_pred)
print('데이터', newdata, ', 판정 결과:', dog_classes[y_pred[0]])

from sklearn.datasets import load_iris

iris = load_iris()
iris.data[:3]     # 최초 3개 데이터의 값을 출력

print(type(iris))

print('iris 데이터의 형태:', iris.data.shape)

print('iris 데이터의 속성들:', iris.feature_names)

print('iris 데이터의 레이블:', iris.target)

import pandas as pd

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = pd.Series(iris.target)
iris_df.head()

# 9.9 k-NN 활용 예제 - 붓꽃 데이터로 학습하기

iris_df.info()
iris_df.describe()
iris_df['target'].value_counts()
iris_df.values

X = iris_df.iloc[:, :4]
y = iris_df.iloc[:, -1]

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

def iris_knn(X, y, k):
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3)
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)

k = 3
scores = iris_knn(X, y, k)
print('n_neighbors가 {0:d}일때 정확도: {1:.3f}'.format(k, scores))


# 9.10 새로운 꽃에 대해서 모델을 적용하고 분류해 보자

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

k = 3
iris = load_iris()
knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(iris.data, iris.target)

classes = {0:'setosa', 1:'versicolor', 2:'virginica'}

# 새로운 데이터를 제시해 보자.
X = [[4, 2, 1.3, 0.4],
     [4, 3, 3.2, 2.2]]
y = knn.predict(X)

print('{} 특성을 가지는 품종: {}'.format(X[0], classes[y[0]]))
print('{} 특성을 가지는 품종: {}'.format(X[1], classes[y[1]]))

y_pred_all = knn.predict(iris.data)
scores = metrics.accuracy_score(iris.target, y_pred_all)
print('n_neighbors가 {0:d}일때 정확도: {1:.3f}'.format(k, scores))

plt.hist2d(iris.target, y_pred_all, bins = (3,3), cmap = plt.cm.jet)
plt.xlabel('actual'), plt.ylabel('predict')
plt.colorbar()

from sklearn.metrics import confusion_matrix
conf_result = confusion_matrix(iris.target,y_pred_all)
print(conf_result)

from sklearn.metrics import confusion_matrix
conf_result = confusion_matrix(iris.target,y_pred_all)
print(np.flip(conf_result, axis = 0))


