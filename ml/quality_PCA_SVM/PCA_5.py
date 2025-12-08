import numpy as np

np.random.seed(0)

n_total = 1000  # число образов выборки
n_features = 200  # число признаков

table = np.zeros(shape=(n_total, n_features))

for _ in range(100):
    i, j = np.random.randint(0, n_total), np.random.randint(0, n_features)
    table[i, j] = np.random.randint(1, 10)

# матрицу table не менять

# здесь продолжайте программу

F = (1 / table.shape[0]) * table.T @ table  # матрица грамма
L, W = np.linalg.eig(F)  # собственные числа и вектора
WW = sorted(
    zip(L, W), key=lambda x: x[0], reverse=True
)  # сортировка векторов относительно чисел
WW = np.array([w[1] for w in WW])  # отсортированная матрица векторов

data_x = table @ WW.T  # новое признаковое пространство
data_x = table @ WW.T[:, : (L > 0.01).sum()]  # отбросиили не информативные признаки
