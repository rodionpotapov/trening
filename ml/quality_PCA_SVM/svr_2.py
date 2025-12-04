import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV



def func(x):
    return np.sin(0.5*x) + 0.2 * np.cos(2*x) - 0.1 * np.sin(4 * x) + 3


# обучающая выборка
coord_x = np.expand_dims(np.arange(-4.0, 6.0, 0.1), axis=1)
coord_y = func(coord_x).ravel()

# здесь продолжайте программу
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
C = [0,0.5,1]
clf = GridSearchCV(kernel,C)
x_train = coord_x[::3]
y_train = coord_y[::3]
svr = svm.SVR(kernel='rbf',degree=11,
        gamma="scale",
        coef0=0.0,
        tol=1e-5,
        C=1,
        epsilon=0.1,
        shrinking=True,
        cache_size=200,
        verbose=False,
        max_iter=-1,)
svr.fit(x_train, y_train)
predict = svr.predict(coord_x)
Q = np.sum((predict - coord_y)**2)
print(Q)

plt.figure(figsize=(8, 5))
# истинная функция
plt.plot(coord_x, coord_y, label="true func", color="black", linewidth=2)
# прогноз SVR
plt.plot(coord_x, predict, label="SVR prediction", color="red")
# обучающие точки
plt.scatter(x_train, y_train, color="blue", s=25, label="train points")

# опорные векторы регрессии (особенно те, что вне ε-трубки)
sv_idx = svr.support_
plt.scatter(x_train[sv_idx], y_train[sv_idx],
            s=80, facecolors='none', edgecolors='green',
            label="support vectors")

plt.legend()
plt.title("SVR с RBF-ядром")
plt.tight_layout()
plt.show()