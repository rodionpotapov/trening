import numpy as np

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return -0.5 * x ** 2 + 0.1 * x ** 3 + np.cos(3 * x) + 7


def w0(w):
    return np.concatenate(([0], w[1:]))

# функция потерь
def loss(w, x, y):
    return (model(w, x) - y) ** 2


# производная функции потерь
def model(w, x):
    return x @ w

def dL(w, x, y):
    r = x @ w - y
    return 2 * r * x


coord_x = np.arange(-4.0, 6.0, 0.1)
coord_y = func(coord_x)

N = 5 # сложность модели (полином степени N-1)
lm_l1 = 2.0 # коэффициент лямбда для L1-регуляризатора
sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001, 0.000002]) # шаг обучения для каждого параметра w0, w1, w2, w3, w4
w = np.zeros(N) # начальные нулевые значения параметров модели
n_iter = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления скользящего экспоненциального среднего
batch_size = 20 # размер мини-батча (величина K = 20)

X = np.array([[1,x,x**2,x**3,x**4] for x in coord_x])
Y = np.array(coord_y)

Qe = 0
np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел

for i in range(n_iter):
    k = np.random.randint(0, sz - batch_size - 1)  # sz - размер выборки (массива coord_x)
    xk = X[k:k + batch_size]
    yk = Y[k:k + batch_size]
    Qk = np.mean([loss(w, xk[i], yk[i]) for i in range(batch_size)])
    # средний градиент по батчу
    dQk = np.mean([dL(w, xk[i], yk[i]) for i in range(batch_size)], axis=0)

    w -= eta * (dQk + lm_l1 * np.sign(w0(w)))
    Qe = lm * Qk + (1 - lm) * Qe

Q = np.mean((X @ w - Y) ** 2)