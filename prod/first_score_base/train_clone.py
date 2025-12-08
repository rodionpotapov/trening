import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, roc_auc_score

data = pd.read_csv("/Users/rodion/PycharmProjects/trening/pandas/datasets/scoring.csv")

X = data.drop(columns=["default"]).values  # признаки
y = data["default"].values  # целевая переменная (0/1)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# пайплайн: сначала стандартизация, потом лог регрессия
pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "logreg",
            LogisticRegression(
                penalty="l2",
                class_weight="balanced",  # если классы несбалансированы
                max_iter=1000,  # дать сходимости время
                solver="liblinear",  # для небольших размерностей хорошо работает
            ),
        ),
    ]
)

param_grid = {
    "logreg__C": np.logspace(-3, 3, 7),  # от 0.001 до 1000
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="f1",  # или "roc_auc"
    cv=5,
    n_jobs=-1,
)

grid.fit(x_train, y_train)

print("Лучшие параметры:", grid.best_params_)
print("Лучшая метрика (cv):", grid.best_score_)

best_model = grid.best_estimator_

y_pred = best_model.predict(x_test)
y_proba = best_model.predict_proba(x_test)[:, 1]

print("F1:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))
