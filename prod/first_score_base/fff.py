import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# 1. Загрузка данных
data = pd.read_csv("/Users/rodion/PycharmProjects/trening/pandas/datasets/scoring.csv")
print(data.head(5))

# 2. Разделение на признаки и целевую переменную
X = data.drop(columns=["default"]).values
y = data["default"].values

# 3. Разбиваем данные на train и test
# stratify=y важно из-за сильного дисбаланса классов
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

# 4. Базовая модель случайного леса
# class_weight='balanced' компенсирует дисбаланс классов
base_model = RandomForestClassifier(
    class_weight="balanced",
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
)

# 5. Подбор гиперпараметров через GridSearchCV
# Сетка небольшая, чтобы не жестко тормозить, но что-то поискать
param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [None, 5, 10],
    "min_samples_leaf": [1, 5, 10],
}

grid = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring="f1",  # баланс precision/recall
    cv=5,
    n_jobs=-1,
    verbose=1,
)

grid.fit(x_train, y_train)

print("Лучшие параметры:", grid.best_params_)

# 6. Лучшая модель из GridSearch
best_model = grid.best_estimator_

# 7. Подбор оптимального порога по F1 на train
proba_train = best_model.predict_proba(x_train)[:, 1]

best_threshold = 0.5
best_f1 = 0.0

# сканируем пороги от 0.1 до 0.9
thresholds = np.linspace(0.1, 0.9, 81)

for t in thresholds:
    y_pred_train = (proba_train > t).astype(int)
    f1 = f1_score(y_train, y_pred_train)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(
    f"Оптимальный порог по F1 на train: {best_threshold:.3f}, F1(train) = {best_f1:.4f}"
)

# 8. Оценка на тестовой выборке с найденным порогом
proba_test = best_model.predict_proba(x_test)[:, 1]
y_pred = (proba_test > best_threshold).astype(int)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Отказано: {y_pred.mean() * 100:.2f}%")
print(f"Точность: {precision * 100:.2f}%")
print(f"Полнота: {recall * 100:.2f}%")
print(f"F1: {f1 * 100:.2f}%")

# 9. Смотрим важности признаков
feature_importances = best_model.feature_importances_
print("Важности признаков:", feature_importances)
