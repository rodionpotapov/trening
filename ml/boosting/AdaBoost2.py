import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


np.random.seed(0)
n_feature = 2

# исходные параметры для формирования образов обучающей выборки
r1 = 0.7
D1 = 3.0
mean1 = [3, 7]
V1 = [[D1 * r1 ** abs(i - j) for j in range(n_feature)] for i in range(n_feature)]

r2 = 0.5
D2 = 2.0
mean2 = [4, 2]
V2 = [[D2 * r2 ** abs(i - j) for j in range(n_feature)] for i in range(n_feature)]

# моделирование обучающей выборки
N1, N2 = 1000, 1200
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N1) * -1, np.ones(N2)])

x_train, x_test, y_train, y_test = train_test_split(
    data_x, data_y, random_state=123, test_size=0.3, shuffle=True
)

# здесь продолжайте программу
w = np.ones(len(x_train)) / len(x_train)
T = 100
max_depth = 3

alfa = []
alg = []

for i in range(T):
    b_t = DecisionTreeClassifier(criterion="gini", max_depth=max_depth)
    alg.append(b_t.fit(x_train, y_train, sample_weight=w))
    pre = alg[i].predict(x_train)

    N = np.sum(np.abs(y_train - pre) / 2 * w)

    alfa.append(0.5 * np.log((1 - N) / N))

    w = w * np.exp(-1 * alfa[i] * y_train * pre)
    w = w / np.sum(w)

predict = alfa[0] * alg[0].predict(x_test)

for i in range(1, T):
    predict += alfa[i] * alg[i].predict(x_test)

predict = np.sign(predict)
Q = np.sum(predict != y_test)
print(Q)
