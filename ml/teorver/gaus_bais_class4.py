import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# исходные параметры распределений трех классов
r1 = 0.7
D1 = 3.0
mean1 = [1, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 2.0
mean2 = [-3, -1]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

r3 = 0.3
D3 = 1.0
mean3 = [1, 2]
V3 = [[D3, D3 * r3], [D3 * r3, D3]]

# моделирование обучающей выборки
N = 10000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T
x3 = np.random.multivariate_normal(mean3, V3, N).T

x_train = np.hstack([x1, x2, x3]).T
y_train = np.hstack([np.zeros(N), np.ones(N), np.ones(N) * 2])

# mm1 = np.mean(x1.T, axis=0)
# mm2 = np.mean(x2.T, axis=0)
# mm3 = np.mean(x3.T, axis=0)
#
# a = (x1.T - mm1).T
# VV1 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
#                 [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])
#
# a = (x2.T - mm2).T
# VV2 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
#                 [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])
#
# a = (x3.T - mm3).T
# VV3 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
#                 [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])


asf = np.cov(mean1, rowvar=True, ddof=None, bias=False, fweights=None, aweights=None)
# m — данные. 1D или 2D массив. При rowvar=True: строки — переменные (M), столбцы — наблюдения (N) → форма (M, N). При rowvar=False: переменные по столбцам → форма (N, M).
# y — второй набор данных той же формы, что и m; если задан, ковариации считаются и между наборами.
# rowvar — ориентация переменных. True: переменные по строкам. False: переменные по столбцам.
# bias — нормировка. False: деление на N−1 (несмещённая). True: деление на N (смещённая). Если указан ddof, он имеет приоритет.
# ddof — число, задающее делитель N−ddof (или сумма весов−ddof при взвешивании).
# fweights — целочисленные неотрицательные частотные веса длины N (как «повторы» наблюдений).
# aweights — вещественные неотрицательные «надёжностные» веса длины N. Если заданы и fweights, и aweights, эффективный вес — их произведение; нормировка по сумме весов (или минус ddof).
# Возврат — ковариационная матрица размера P×P, где P — число переменных (для одной переменной может вернуть скаляр — дисперсию).

mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)
mm3 = np.mean(x3.T, axis=0)

VV1 = np.cov(x1, ddof=0)
VV2 = np.cov(x2, ddof=0)
VV3 = np.cov(x3, ddof=0)

# параметры для гауссовского байесовского классификатора
Py1, Py2, Py3 = 0.2, 0.5, 0.3
L1, L2, L3 = 1, 1, 1

# здесь продолжайте программу
predict = []
b = lambda x, v, m, l, py: np.log(l * py) - 0.5 * (x - m) @ np.linalg.inv(v) @ (x - m).T - 0.5 * np.log(
    np.linalg.det(v))
for i in range(len(x_train)):
    a = np.argmax(
        [b(x_train[i], VV1, mm1, L1, Py1), b(x_train[i], VV2, mm2, L2, Py2), b(x_train[i], VV3, mm3, L3, Py3)])
    predict.append(a)

Q = np.sum(predict != y_train)

# --- предвычисления для стабильности/скорости ---
means = [mm1, mm2, mm3]
invs = [np.linalg.inv(VV1), np.linalg.inv(VV2), np.linalg.inv(VV3)]
logd = [np.log(np.linalg.det(VV1)), np.log(np.linalg.det(VV2)), np.log(np.linalg.det(VV3))]
priors = [Py1, Py2, Py3]
losses = [L1, L2, L3]


def score(x, k):
    d = x - means[k]
    return np.log(losses[k] * priors[k]) - 0.5 * d @ invs[k] @ d - 0.5 * logd[k]


# --- решётка и классификация каждой точки решётки ---
pad = 1.0
x_min, y_min = x_train.min(axis=0) - pad
x_max, y_max = x_train.max(axis=0) + pad

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400),
)
grid = np.c_[xx.ravel(), yy.ravel()]

S = np.stack([
    np.array([score(p, 0) for p in grid]),
    np.array([score(p, 1) for p in grid]),
    np.array([score(p, 2) for p in grid]),
], axis=1)
zz = np.argmax(S, axis=1).reshape(xx.shape)

# --- отрисовка областей решений и обучающих точек ---
cmap_bg = ListedColormap(["#fde2e2", "#e1f0ff", "#e6f4ea"])
colors_pts = ["#d81b60", "#1e88e5", "#43a047"]

plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5, 2.5], cmap=cmap_bg, alpha=0.35)

plt.scatter(x1[0], x1[1], s=10, c=colors_pts[0], label="class 0")
plt.scatter(x2[0], x2[1], s=10, c=colors_pts[1], label="class 1")
plt.scatter(x3[0], x3[1], s=10, c=colors_pts[2], label="class 2")

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Gaussian Bayes — decision regions")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
