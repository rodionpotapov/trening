import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn import svm
import joblib
from sklearn.model_selection import GridSearchCV
import numpy as np


data = pd.read_csv("/Users/rodion/PycharmProjects/trening/pandas/datasets/scoring.csv")
print(data.head(5))


X = data.drop(columns=["default"]).values  # данные из датасета для обучения
Y = data["default"].values  # данные рефернс

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2
)  # hold_out: 80% - train, 20% - валидация

model = LogisticRegression(
    class_weight="balanced",
    penalty="l2",
)  # увеличение штрафа за неверную классификацию


model.fit(x_train, y_train)

y_pred = model.predict(x_test)  # предсказание модели

"""метрики"""
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Отказано: {y_pred.mean() * 100:.2f}%")
print(f"Точность: {precision * 100:.2f}%")
print(f"Полнота: {recall * 100:.2f}%")


joblib.dump(
    model, "../models/model.pkl"
)  # бинарный файл с моделью которая получилась после обучения
