import numpy as np

np.random.seed(0)

# исходные параметры распределений двух классов
mean1 = [1, -2]
mean2 = [1, 3]
r = 0.7
D = 2.0
V = [[D, D * r], [D * r, D]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V, N).T
x2 = np.random.multivariate_normal(mean2, V, N).T

x_train = np.hstack([x1, x2]).T
y_train = np.hstack([np.ones(N) * -1, np.ones(N)])

# вычисление оценок МО и ковариационной матрицы
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

a = np.hstack([(x1.T - mm1).T, (x2.T - mm2).T])
VV = np.array([[np.dot(a[0], a[0]) / (2*N), np.dot(a[0], a[1]) / (2*N)],
                [np.dot(a[1], a[0]) / (2*N), np.dot(a[1], a[1]) / (2*N)]])

# здесь продолжайте программу
lm = 1
py = 0.5
predict = []
b = lambda x, v, m, l, py: np.log(l * py) - 0.5 * m.T @ np.linalg.inv(v) @ m + x.T @ np.linalg.inv(v) @ m
for i in range(len(x_train)):
    a = np.argmax([b(x_train[i], VV, mm1, lm, py), b(x_train[i], VV, mm2, lm, py)]) * 2 -1   # классификатор
    predict.append(a)

Q = np.sum(predict != y_train)
print(Q)