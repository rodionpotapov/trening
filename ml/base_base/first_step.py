import numpy as np

#модель: y = w_0 + w_1 * x_i +eps


np.random.seed(0) # псевдослучайные числа образуют одну и ту же последовательность
x = np.arange(-1.0, 1.0, 0.1) # аргумент [-1; 1] с шагом 0,1

size_train = len(x)  # размер выборки
w = [0.5, -0.3]  # коэффициенты модели
model_a = lambda m_x, m_w: (m_w[1] * m_x + m_w[0])  # модель
loss = lambda ax, y: (ax - y) ** 2 # квадратическая функция потерь

y = model_a(x, w) + np.random.normal(0, 0.1, len(x)) # целевые значения

Q = loss(model_a(x,w),y).mean() #СЭР
