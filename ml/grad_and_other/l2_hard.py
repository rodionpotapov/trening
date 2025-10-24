import numpy as np


# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 2.5


def w0(w):
    return np.concatenate(([0], w[1:]))


def loss(w, x, y):
    return (x @ w - y) ** 2


def dloss(w, x, y):
    return 2 * (x @ w - y) * x.T


coord_x = np.arange(-4.0, 6.0, 0.1)  # значения по оси абсцисс [-4; 6] с шагом 0.1
coord_y = func(coord_x)  # значения функции по оси ординат

N = 5  # сложность модели (полином степени N-1)
lm_l2 = 2  # коэффициент лямбда для L2-регуляризатора
sz = len(coord_x)  # количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001, 0.000002])  # шаг обучения для каждого параметра w0, w1, w2, w3, w4
w = np.zeros(N)  # начальные нулевые значения параметров модели
n_iter = 500  # число итераций алгоритма SGD
lm = 0.02  # значение параметра лямбда для вычисления скользящего экспоненциального среднего
batch_size = 20  # размер мини-батча (величина K = 20)

X = np.array([[a ** n for n in range(N)] for a in coord_x])
Y = np.array(coord_y)

Qe = np.mean(loss(w, X, Y))  # начальное значение среднего эмпирического риска
np.random.seed(0)  # генерация одинаковых последовательностей псевдослучайных чисел

for _ in range(n_iter):
    k = np.random.randint(0, sz - batch_size - 1)

    Qk = sum([loss(w, X[i], Y[i]) for i in range(k, k + batch_size)]) / batch_size
    Qe = lm * Qk + (1 - lm) * Qe

    dQk = sum([dloss(w, X[i], Y[i]) for i in range(k, k + batch_size)]) / batch_size
    w -= eta * (dQk + lm_l2 * w0(w)) #без w0 коэфицента

Q = np.mean(loss(w, X, Y))