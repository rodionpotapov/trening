import numpy as np
from sklearn.tree import DecisionTreeRegressor


x = np.arange(-3, 3, 0.1).reshape(-1, 1)
y = 2 * np.cos(x) + 0.5 * np.sin(2 * x) - 0.2 * np.sin(4 * x)


s = np.array(y.ravel())

algs = []
T = 6
max_depth = 3


# продолжите программу
for i in range(T):
    b_t = DecisionTreeRegressor(max_depth=max_depth)
    algs.append(b_t.fit(x, s))
    pred = algs[-1].predict(x)
    print(np.mean(pred))

    s -= pred

yy = 0
for n in range(0, T):
    yy += algs[n].predict(x)

QT = np.mean((y.ravel() - yy) ** 2)
print(QT)
