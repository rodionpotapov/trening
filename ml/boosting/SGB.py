import numpy as np
from sklearn.tree import DecisionTreeRegressor


np.random.seed(42)

x = np.arange(-3, 3, 0.1).reshape(-1, 1)
y = (
    2 * np.cos(x)
    + 0.5 * np.sin(2 * x)
    - 0.2 * np.sin(4 * x)
    + 0.05 * np.random.randn(len(x), 1)
)

y = y.ravel()

algs = []
T = 10
max_depth = 3
N = len(y)
lr = 0.2
subsample = 0.6
min_leaf = 3


def loss(f, y):
    return 0.5 * np.sum((f - y) ** 2)


def df(f, y):
    return f - y


F0 = np.array([0] * N)
F = F0.copy()

m = int(N * subsample)

for i in range(T):
    idx = np.random.choice(N, size=m, replace=False)  # рандомный батч
    r = y - F
    b_t = DecisionTreeRegressor(
        max_depth=max_depth, min_samples_leaf=min_leaf, random_state=42 + i
    )
    b_t.fit(x[idx], r[idx])
    algs.append(b_t)
    F = F + lr * b_t.predict(x)

QT = np.mean((y - F) ** 2)
print(QT)
