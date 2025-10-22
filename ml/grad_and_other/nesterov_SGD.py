import numpy as np

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return -0.7 * x - 0.2 * x ** 2 + 0.05 * x ** 3 - 0.2 * np.cos(3 * x) + 2

def Qk(w,x,y):
    return np.mean(np.square(x @ w - y))

def dQk(w, x, y):
    K = x.shape[0]
    return (2.0 / K) * (x.T @ (x @ w - y))



coord_x = np.arange(-4.0, 6.0, 0.1) # значения по оси абсцисс [-4; 6] с шагом 0.1
coord_y = func(coord_x) # значения функции по оси ординат

X = np.array([[1,x,x**2, x**3] for x in coord_x])
Y = np.array(coord_y)
sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001]) # шаг обучения для каждого параметра w0, w1, w2, w3
w = np.array([0., 0., 0., 0.]) # начальные значения параметров модели
N = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления скользящего экспоненциального среднего
batch_size = 20 # размер мини-батча (величина K = 20)
gamma = 0.8 # коэффициент гамма для вычисления импульсов Нестерова
v = np.zeros(len(w))  # начальное значение [0, 0, 0, 0]

Qe = 0  # начальное значение среднего эмпирического риска
np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел

for i in range(N):
    k = np.random.randint(0, sz - batch_size - 1)  # sz - размер выборки (массива coord_x)
    xk = X[k:k+batch_size]
    yk = Y[k:k+batch_size]
    Qe = lm * Qk(w,xk,yk) + (1 - lm) * Qe
    v = gamma * v + (1 - gamma) * eta * dQk(w-gamma * v, xk,yk)
    w -= v

Q = Qk(w,X,Y)