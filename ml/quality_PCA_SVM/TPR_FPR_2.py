import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

np.random.seed(0)

# исходные параметры распределений классов
r1 = -0.2
D1 = 3.0
mean1 = [1, -5]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 2.0
mean2 = [-1, -2]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# моделирование обучающей выборки
N1 = 1000
N2 = 1000
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N1) * -1, np.ones(N2)])

x_train, x_test, y_train, y_test = train_test_split(
    data_x, data_y, random_state=123, test_size=0.5, shuffle=True
)

# clf = svm.SVC(kernel="linear")
# clf.fit(x_train, y_train)
# w = np.array([clf.intercept_[0], *clf.coef_[0]])
# x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))


range_t = np.arange(5.7, -7.8, -0.1)
FPR = []
TPR = []

fg = svm.SVC(kernel="rbf")
fg.fit(x_train, y_train)
score = fg.decision_function(
    x_test
)  # решающая функция как в линейном ядре w0 + w1x + w2x + ..

for t in range_t:
    predict = np.sign(score - t)  # список меток классов
    TP = sum([predict[i] == y_test[i] and predict[i] == 1 for i in range(len(predict))])
    TN = sum(
        [predict[i] == y_test[i] and predict[i] == -1 for i in range(len(predict))]
    )
    FP = sum([predict[i] != y_test[i] and predict[i] == 1 for i in range(len(predict))])
    FN = sum(
        [predict[i] != y_test[i] and predict[i] == -1 for i in range(len(predict))]
    )
    FPR.append(FP / (FP + TN))
    TPR.append(TP / (TP + FN))

FPR = np.array(FPR)
TPR = np.array(TPR)

# сортируем точки по FPR, чтобы можно было численно интегрировать
order = np.argsort(FPR)
FPR_sorted = FPR[order]
TPR_sorted = TPR[order]

# численное значение ROC AUC (интеграл по трапециям)
roc_auc = np.trapz(TPR_sorted, FPR_sorted)
print("ROC AUC:", roc_auc)

# построение ROC-кривой
plt.plot(FPR_sorted, TPR_sorted, label=f"ROC-кривая (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "--", label="случайная модель")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC-кривая для SVM (RBF)")
plt.legend()
plt.grid(True)
plt.show()
