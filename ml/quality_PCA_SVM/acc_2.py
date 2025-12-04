import numpy as np
from sklearn.model_selection import train_test_split


def loss(w, x, y):
    return np.exp(-w.T @ x * y)


def df(w, x, y):
    M = y * (w @ x)
    return - y * x * np.exp(-M)


np.random.seed(0)

# исходные параметры распределений двух классов
r1 = 0.4
D1 = 2.0
mean1 = [1, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 3.0
mean2 = [2, 3]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T

data_x = np.array([[1, x[0], x[1]] for x in np.hstack([x1, x2]).T])
data_y = np.hstack([np.ones(N) * -1, np.ones(N)])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123, test_size=0.3, shuffle=True)

n_train = len(x_train)  # размер обучающей выборки
w = np.array([0.0, 0.0, 0.0])  # начальные весовые коэффициенты
nt = np.array([0.5, 0.01, 0.01])  # шаг обучения для каждого параметра w0, w1, w2
N = 500  # число итераций алгоритма SGD
batch_size = 10  # размер мини-батча (величина K = 10)

# здесь продолжайте программу
for i in range(N):
    k = np.random.randint(0, n_train - batch_size - 1)  # n_train - размер выборки (массива x_train)
    xi = x_train[k:k + batch_size]
    yi = y_train[k:k + batch_size]
    dQk = sum([df(w,xi[i],yi[i]) for i in range(batch_size)]) / batch_size
    w -= nt * dQk

mrgs = sorted(y_test * np.dot(x_test,w))
scores = x_test @ w
y_pred = np.sign(scores)
acc = np.mean(y_pred == y_test)
print(acc)