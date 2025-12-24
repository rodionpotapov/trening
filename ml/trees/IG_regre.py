import numpy as np

x = np.arange(-2, 3, 0.1)
y = -x + 0.2 * x**2 - 0.5 * np.sin(4 * x) + np.cos(2 * x)

# здесь продолжайте программу
t = 0
data_l = y[x < t]
data_r = y[x >= t]


def ax(y):
    return np.mean(y)


def mse(y, y_s):
    return np.sum((ax(y) - y_s) ** 2)


H0 = mse(y, y)
H1 = mse(data_l, data_l)
H2 = mse(data_r, data_r)

IG = H0 - ((len(data_l) / len(y)) * H1 + (len(data_r) / len(y)) * H2)
print(IG)
