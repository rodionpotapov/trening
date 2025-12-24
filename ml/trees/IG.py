import numpy as np

np.random.seed(0)
X = np.random.randint(0, 2, size=200)

# здесь продолжайте программу
t = 150


def cnt(x, ui):
    return np.sum(x == ui) / len(x)


def geany(x, ui_0, ui_1):
    return 1 - (cnt(x, ui_0) ** 2 + cnt(x, ui_1) ** 2)


S0 = geany(X, 0, 1)
S_left = geany(X[:t], 0, 1)  # 150
S_right = geany(X[t:], 0, 1)  # 50

IG = S0 - (S_left * t / 200 + S_right * (200 - t) / 200)
