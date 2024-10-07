r"""
KNearestNeighbours
===

This module contains an implementation of a KNN class for our Machine Learning project.

This module exports:
  - KNN class

Usage
-----
To create an instance of the KNN class, do::

  >>> model = KNN(k)

Constructor parameters are (in order):
  - `k` number of neighbours to find when predicting

To train a created model, do::

  >>> model.fit(x_train, y_train)

To predict labels by getting the K nearest neighbours, do::

  >>> model.predict(x_test)

"""
from scipy.spatial import KDTree
from scipy.stats import mode
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

### KFold for KNN
def KNN_Kfold(X, Y):
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    results = np.zeros((10, 4))

    i = 0
    for train_index, test_index in kfold.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        knn = NearestNeighbor(k=2)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)

        report = classification_report(y_test, y_pred, target_names=["Negative", "Positive"], output_dict=True)
        a = report['accuracy']
        p = report['weighted avg']['precision']
        r = report['weighted avg']['recall']
        f = report['weighted avg']['f1-score']
        results[i] = [a,p,r,f]
        i += 1
    
    iterations = range(results.shape[0])

    plt.plot(iterations, results[:, 0], label='Accuracy', color='red', linewidth=1)
    plt.plot(iterations, results[:, 1], label='Precision', color='blue', linewidth=1)
    plt.plot(iterations, results[:, 2], label='F1 Score', color='green', linewidth=1)
    plt.plot(iterations, results[:, 3], label='Recall', color='purple', linewidth=1)

    plt.title('10-fold Cross Validation Performance Metrics of KNN')
    plt.xlabel('Fold')
    plt.ylabel('Scores')
    plt.legend(loc="lower right")
    plt.show()

### MANUAL IMPLEMENTATION OF KNN
class NearestNeighbor:
    def __init__(self, k=5):
        self.k = k
        self.tree = None
        self.y_train = None
    
    def fit(self, X_train, y_train):
        self.tree = KDTree(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        d, indices = self.tree.query(X_test, k=self.k)
        labels = self.y_train[indices]

        if self.k == 1:
            return labels.flatten()
        else:
            common_label = mode(labels, axis=1)
            return common_label.mode.flatten()

def run_KNN(x_train, x_test, y_train, y_test, k):
    knn = NearestNeighbor(k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    report = classification_report(y_test, y_pred, target_names=["Negative", "Positive"], output_dict=True)
    f1 = report['weighted avg']['f1-score']
    return f1
