import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split

np.random.seed(0)

# исходные параметры распределений классов
r1 = 0.6
D1 = 3.0
mean1 = [1, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 2.0
mean2 = [-2, -1]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# моделирование обучающей выборки
N = 500
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N) * -1, np.ones(N)])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123,test_size=0.4, shuffle=True)

clf = svm.SVC(
    C=1.0,                  # сила штрафа за ошибки (↑C → меньше ошибок на train, но больше риск переобучения)
    kernel="rbf",           # тип ядра: 'linear', 'poly', 'rbf', 'sigmoid'
    degree=3,               # степень полинома (только для kernel='poly')
    gamma="scale",          # радиус влияния точки для rbf/poly/sigmoid ('scale' подбирается автоматически)
    coef0=0.0,              # сдвиг в полиномиальном/сигмоидальном ядре
    shrinking=True,         # использовать ускоренный алгоритм SMO
    probability=False,      # нужно ли потом уметь predict_proba (дороже по времени)
    tol=1e-6,               # точность/критерий остановки оптимизации
    cache_size=200,         # размер кэша для ядра (МБ)
    class_weight=None,      # веса классов, напр. 'balanced' при дисбалансе
    verbose=True,           # печатать прогресс обучения
    max_iter=-1,            # ограничение по кол-ву итераций (-1 = без ограничения)
    decision_function_shape="ovr",  # схема для мультикласса: 'ovr' (по умолчанию) или 'ovo'
    break_ties=False,       # как разруливать равные значения при 'ovr'
    random_state=None       # фиксировать случайность (если нужно воспроизводимо)
)
clf.fit(x_train, y_train)
predict = clf.predict(x_test)

Q = np.sum(predict != y_test)
print(Q)



xx, yy = np.meshgrid(
    np.linspace(data_x[:, 0].min() - 1, data_x[:, 0].max() + 1, 400),
    np.linspace(data_x[:, 1].min() - 1, data_x[:, 1].max() + 1, 400),
)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = clf.predict(grid).reshape(xx.shape)

plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, Z, alpha=0.25, levels=[-1, 0, 1], colors=["#ffaaaa", "#aaaaff"])

# точки train
plt.scatter(
    x_train[y_train == -1, 0],
    x_train[y_train == -1, 1],
    c="red",
    edgecolor="k",
    label="class -1",
    s=30,
)
plt.scatter(
    x_train[y_train == 1, 0],
    x_train[y_train == 1, 1],
    c="blue",
    edgecolor="k",
    label="class +1",
    s=30,
)

# сами опорные векторы
sv = clf.support_vectors_
plt.scatter(
    sv[:, 0],
    sv[:, 1],
    s=120,
    facecolors="none",
    edgecolors="k",
    linewidths=1.5,
    label="support vectors",
)

plt.legend()
plt.title("SVM: разделяющая поверхность и опорные векторы")
plt.xlabel("x1")
plt.ylabel("x2")
plt.tight_layout()
plt.show()

# 2) ВЫЧЛЕНИТЬ 'ПРОБЛЕМНЫЕ' / ВОЗМОЖНЫЕ ВЫБРОСЫ В КОДЕ
# В sklearn SVC dual_coef_ содержит yi * alpha_i только для support vectors.
# Если alpha_i ~= C -> эта точка либо внутри зазора, либо вообще по ошибке (то есть "тяжёлая").
C = clf.C
sv_idx = clf.support_                # индексы support vectors в x_train
alphas = np.abs(clf.dual_coef_[0])   # |alpha_i| для support vectors

# считаем, что "подозрительные" те, у кого alpha очень близко к C
eps = 1e-6
outlier_mask = alphas >= (C - eps)
outlier_indices_in_train = sv_idx[outlier_mask]
outlier_points = x_train[outlier_indices_in_train]
outlier_labels = y_train[outlier_indices_in_train]

print("Индексы подозрительных точек в train:", outlier_indices_in_train)
print("Координаты этих точек:")
print(outlier_points)
print("Их истинные метки:")
print(outlier_labels)