import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from knn import KNN
from svm import SVM
from logireg import LR
from dtree import DT

def get_bootstrapped_dataset(x, y):
    rows = x.shape[0] # n of samples
    indexes = np.random.choice(rows, size=rows, replace=True)
    x_boot = x[indexes]
    y_boot = y[indexes]
    return x_boot, y_boot

def display_results(results):
    iterations = range(results.shape[0])

    plt.plot(iterations, results[:, 0], label='Accuracy', color='red', linewidth=1)
    plt.plot(iterations, results[:, 1], label='Precision', color='blue', linewidth=1)
    plt.plot(iterations, results[:, 2], label='F1 Score', color='green', linewidth=1)
    plt.plot(iterations, results[:, 3], label='Recall', color='purple', linewidth=1)

    plt.title('Bootstrapping Performance Metrics')
    plt.xlabel('Iteration')
    plt.ylabel('Scores')
    plt.legend(loc="lower right")
    plt.show()


def test_knn_bootstrapped(x, y, iterations = 10):
    model = KNN()
    results = np.zeros((iterations, 4))
    for i in range(iterations):
        x_boot, y_boot = get_bootstrapped_dataset(x, y)
        model.fit(x_boot, y_boot)
        y_pred = model.predict(x_boot)

        acc = accuracy_score(y_boot, y_pred)
        prec = precision_score(y_boot, y_pred)
        f1 = f1_score(y_boot, y_pred)
        rec = recall_score(y_boot, y_pred)

        results[i] = [acc, prec, f1, rec]
    display_results(results)
