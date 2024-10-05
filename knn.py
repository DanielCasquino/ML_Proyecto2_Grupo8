from scipy.spatial import KDTree
from scipy.stats import mode
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd

### KNN implementation with K-fold with sklearn
def KNN_Kfold(X, Y, k):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    knn = KNeighborsClassifier(k)
    y_pred = cross_val_predict(knn, X_scaled, Y, cv=10)
    precision = precision_score(Y, y_pred, pos_label=1)
    recall = recall_score(Y, y_pred, pos_label=1)
    f1 = f1_score(Y, y_pred, pos_label=1)

    scores = cross_val_score(knn, X_scaled, Y, cv=5, scoring='accuracy')
    mean_accuracy = np.mean(scores)
    error = 1 - mean_accuracy

    return y_pred, mean_accuracy, precision, recall, f1, error

### Handy function to calculate the best k
def get_best_k(X, Y, n):
    res = []

    for i in range(1, n+1):
        y_pred, mean_accuracy, precision, recall, f1, error = KNN_Kfold(X, Y, i)
        r = [i, round(mean_accuracy, 3), round(precision, 3), round(recall, 3), round(f1, 3), round(error, 3)]
        res.append(r)

    df = pd.DataFrame(res, columns=['k', 'Mean Accuracy','Precision', 'Recall', 'F1 Score', 'Error'])
    return df

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
        common_label = mode(labels, axis=1)
        return common_label.mode.flatten()
