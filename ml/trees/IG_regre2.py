import numpy as np

x = np.arange(-2, 3, 0.1)
y = -x + 0.2 * x**2 - 0.5 * np.sin(4 * x) + np.cos(2 * x)


# здесь продолжайте программу
def ax(y):
    return np.mean(y)


def mse(y, ys):
    return np.sum((ax(y) - ys) ** 2)


th = -np.inf
ig = -np.inf

for t in x:
    data_l = y[x < t]
    data_r = y[x >= t]

    h0 = mse(y, y)
    hr1 = mse(data_l, data_l)  # MSE на сплите
    hr2 = mse(data_r, data_r)  # MSE на сплите

    IG = h0 - (
        ((len(data_l) / len(y)) * hr1) + ((len(data_r) / len(y)) * hr2)
    )  # инф выйгрыш сплита - ищем макс
    if IG > ig:
        ig = IG
        th = t

print(th, IG)
