# Решающее дерево для задачи регрессии

from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return 0.5 * x**2 - 0.1 * 1 / np.exp(-x) + 0.5 * np.cos(2 * x) - 2.0 + np.exp(x)


x = np.arange(0, np.pi, 0.1).reshape(-1, 1)
y = func(x)

clf = tree.DecisionTreeRegressor(max_depth=4)
clf = clf.fit(x, y)
yy = clf.predict(x)

# tree.plot_tree(clf)
plt.plot(x, y, label="cos(x)")
plt.plot(x, yy, label="DT Regression")
plt.grid()
plt.legend()
plt.title("max_depth=3")
plt.show()
