from scipy.spatial import KDTree
from scipy.stats import mode
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

### Implementacion de knn con k-fold usando sklearn
def KNN_Kfold(X, Y, k):
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree')
    accuracies = cross_val_score(knn, X, Y, cv=10, scoring='accuracy')    
    accuracy = accuracies.mean()
    error = 1 - accuracy
    return accuracy, error
    
### Calcular k-fold a traves de multiples valores de k. El mejor se saca al ojo
def get_best_k(X, Y, n):
    res = []

    for i in range(1, n+1):
        mean_accuracy, mean_error = KNN_Kfold(X, Y, i)
        r = [i, round(mean_accuracy, 2), round(mean_error, 2)]
        res.append(r)

    df = pd.DataFrame(res, columns=['k', 'Mean Accuracy', 'Mean Error'])
    return df

### IMPLEMENTACION DE KNN CON KDTREE
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
