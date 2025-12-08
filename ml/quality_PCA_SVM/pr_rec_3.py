import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

np.random.seed(0)

# исходные параметры распределений классов
r1 = 0.2
D1 = 3.0
mean1 = [2, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 2.0
mean2 = [-1, -1]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# моделирование обучающей выборки
N1 = 2500
N2 = 1500
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N1) * -1, np.ones(N2)])

x_train, x_test, y_train, y_test = train_test_split(
    data_x, data_y, random_state=123, test_size=0.4, shuffle=True
)

model = SVC()

param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf", "poly"],
    "gamma": ["scale", 0.1, 0.01],  # для rbf
}

from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(
    estimator=model,  # любая sklearn-модель (SVC(), KNN(), LogisticRegression(), ...)
    param_grid=param_grid,  # dict или список dict’ов вида:
    # {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
    # переберёт все комбинации
    scoring="recall",  # строка с именем метрики ("accuracy","precision","recall","f1",
    # "roc_auc", "neg_mean_squared_error", ...) или callable
    n_jobs=-1,  # int: 1 = один поток, -1 = все ядра, 2 = два потока и т.д.
    cv=100,  # int: k-fold; или объект CV (StratifiedKFold, GroupKFold, ...);
    # или iterable разбиений
    refit=True,  # bool или имя метрики; True = обучить заново на всех данных
    # с лучшими найденными гиперами
    verbose=1,  # 0 = молчать, 1 = немного логов, 2+ = многословно
    return_train_score=False,  # True — в cv_results_ будут train-метрики тоже
    error_score=np.nan,  # float или "raise": что писать в результаты, если fit упал
)
clf.fit(x_train, y_train)
# w = [clf.intercept_[0], clf.coef_[0][0], clf.coef_[0][1]]
pred = clf.predict(x_test)
red = clf.best_estimator_.predict(
    x_test
)  # refit говоит лучшая комбинация гиперов — вот эта, после подбора гиперов обучи ЛУЧШУЮ модель ещё раз на всех данных.

TP = np.sum((pred == 1) & (y_test == 1))
TN = np.sum((pred == -1) & (y_test == -1))
FP = np.sum((pred == 1) & (y_test == -1))
FN = np.sum((pred == -1) & (y_test == 1))

precision = TP / (TP + TN)
recall = TP / (TP + FN)

F = 2 * precision * recall / (precision + recall)
Fb = (1 + 0.25) * precision * recall / (0.25 * (precision + recall))
print(precision, recall, F, Fb)
