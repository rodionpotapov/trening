# Агломеративная иерархическая кластеризация с помощью пакета Sk-Learn

from itertools import cycle
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt


# функция для отображения дендограммы
def plot_dendrogram(model, **kwargs):
    children = model.children_
    distances = model.distances_
    n_samples = model.labels_.shape[0]

    counts = np.zeros(children.shape[0])
    for i, (a, b) in enumerate(children):
        ca = 1 if a < n_samples else counts[a - n_samples]
        cb = 1 if b < n_samples else counts[b - n_samples]
        counts[i] = ca + cb

    linkage_matrix = np.column_stack([children, distances, counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)


# входные образы для кластеризации
x = [
    (89, 151),
    (114, 120),
    (156, 110),
    (163, 153),
    (148, 215),
    (170, 229),
    (319, 166),
    (290, 178),
    (282, 222),
]

# data_x = [
#     (48, 118),
#     (74, 96),
#     (103, 82),
#     (135, 76),
#     (162, 79),
#     (184, 97),
#     (206, 111),
#     (231, 118),
#     (251, 118),
#     (275, 110),
#     (298, 86),
#     (320, 68),
#     (344, 62),
#     (376, 61),
#     (403, 75),
#     (424, 95),
#     (440, 114),
#     (254, 80),
#     (219, 85),
#     (288, 66),
#     (260, 92),
#     (201, 76),
#     (162, 66),
#     (127, 135),
#     (97, 143),
#     (83, 160),
#     (82, 177),
#     (88, 199),
#     (105, 205),
#     (135, 208),
#     (151, 198),
#     (157, 169),
#     (153, 152),
#     (117, 158),
#     (106, 168),
#     (106, 185),
#     (123, 188),
#     (125, 171),
#     (139, 163),
#     (139, 183),
#     (358, 127),
#     (328, 132),
#     (313, 146),
#     (300, 169),
#     (300, 181),
#     (308, 197),
#     (326, 206),
#     (339, 209),
#     (370, 199),
#     (380, 184),
#     (380, 147),
#     (343, 154),
#     (329, 169),
#     (332, 184),
#     (345, 185),
#     (363, 159),
#     (361, 177),
#     (344, 169),
#     (311, 175),
#     (351, 89),
#     (134, 96),
# ]

x = np.array(x)

NC = 4  # максимальное число кластеров (итоговых)

# агломеративная иерархическая кластеризация
clustering = AgglomerativeClustering(
    n_clusters=NC, linkage="ward", compute_distances=True
)
x_pr = clustering.fit_predict(x)

# отображение результата кластеризации и дендограммы
f, ax = plt.subplots(1, 2)
for c, n in zip(cycle("bgrcmykgrcmykgrcmykgrcmykgrcmykgrcmyk"), range(NC)):
    clst = x[x_pr == n].T
    ax[0].scatter(clst[0], clst[1], s=10, color=c)

plot_dendrogram(clustering, ax=ax[1])
plt.show()
