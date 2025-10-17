import numpy as np

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.1 * x**2 - np.sin(x) + 5.


# здесь объявляйте необходимые функции


coord_x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс [-5; 5] с шагом 0.1
coord_y = func(coord_x) # значения функции по оси ординат

sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001]) # шаг обучения для каждого параметра w0, w1, w2, w3
w = np.array([0., 0., 0., 0.]) # начальные значения параметров модели
N = 200 # число итераций градиентного алгоритма


s = np.array([np.ones_like(coord_x), coord_x, coord_x ** 2, coord_x ** 3]).T

for i in range(N):
    gradient = (2 / sz) * (w @ s.T - coord_y) @ s  # Градиент
    w-=eta*gradient

Q = np.sum((w @ s.T - coord_y) ** 2).mean()