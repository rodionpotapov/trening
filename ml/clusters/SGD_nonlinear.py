import numpy as np


# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.5 * x**2 - 0.1 * 1 / np.exp(-x) + 0.5 * np.cos(2 * x) - 2.0


# здесь объявляйте необходимые функции
def grad(w, x, y):
    return 2 * (w.T @ x - y) * x


def loss(w, x, y):
    return (w.T @ x - y) ** 2


coord_x = np.arange(-5.0, 5.0, 0.1)  # значения по оси абсцисс [-5; 5] с шагом 0.1
coord_y = func(coord_x)  # значения функции по оси ординат

sz = len(coord_x)  # количество значений функций (точек)
eta = np.array(
    [0.01, 0.001, 0.0001, 0.01, 0.01]
)  # шаг обучения для каждого параметра w0, w1, w2, w3, w4
w = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # начальные значения параметров модели
N = 500  # число итераций алгоритма SGD
lm = 0.02  # значение параметра лямбда для вычисления скользящего экспоненциального среднего

Qe = 0  # начальное значение среднего эмпирического риска
np.random.seed(0)  # генерация одинаковых последовательностей псевдослучайных чисел

for i in range(N):
    k = np.random.randint(0, sz - 1)  # sz - размер выборки (массива coord_x)
    obj = coord_x[k]
    tar = coord_y[k]
    xk = np.array([1, obj, obj**2, np.cos(2 * obj), np.sin(2 * obj)])
    w -= eta * grad(w, xk, tar)

    Qe = lm * loss(w, xk, tar) + (1 - lm) * Qe

X = np.column_stack(
    [
        np.ones_like(coord_x),
        coord_x,
        coord_x**2,
        np.cos(2 * coord_x),
        np.sin(2 * coord_x),
    ]
)
Q = np.mean((w @ X.T - coord_y) ** 2)
print(Q)
