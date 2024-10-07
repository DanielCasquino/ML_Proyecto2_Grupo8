r"""
LogisticRegression
===

This module contains an implementation of a logistic regression model class for our Machine Learning project.

This module exports:
  - LR class

Usage
-----
To create an instance of the LR class, do::

  >>> model = LR(alpha, lam)

Constructor parameters are (in order):
  - `alpha` the alpha constant - controls model reaction to slope change
  - `lam` lambda constant, used for regularization

To train a created model, do::

  >>> model.fit(x_train, y_train, epochs)

To predict using the model, do::

  >>> model.predict(x_test)

"""
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

### KFold for KNN
def LR_Kfold(X, Y):
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    results = np.zeros((10, 4))

    i = 0
    for train_index, test_index in kfold.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        lr = LR(alpha=0.00001, lam=0.01)
        lr.fit(x_train, y_train, epochs=5000)
        y_pred = lr.predict(x_test)

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

    plt.title('10-fold Cross Validation Performance Metrics of LR')
    plt.xlabel('Fold')
    plt.ylabel('Scores')
    plt.legend(loc="lower right")
    plt.show()

class LR:
  def __init__(self, alpha, lam):
    self.alpha = alpha
    self.lam = lam
    self.w = None

  def h(self, x):
    return np.dot(x, self.w.T)

  def s(self, x):
    return 1 / (1 + np.exp(-self.h(x)))

  def loss(self, y, y_aprox):
    n = len(y)
    ep = 1e-15
    y_aprox = np.clip(y_aprox,ep,1-ep)
    loss = -1 / n * np.sum(y * np.log(y_aprox) + (1 - y) * np.log(1 - y_aprox)) + self.lam * np.sum(self.w ** 2)
    return loss

  def derivatives(self, x, y):
    s = self.s(x)
    return (np.dot((s - y).T, x) / len(y)) + 2 * self.lam * self.w

  def update(self, d_w):
    self.w -= (self.alpha * d_w)

  def fit(self, x_train, y_train, epochs):
    n= x_train.shape[1]
    self.w = np.zeros(n)
    for _ in range(epochs): 
      y_aprox = self.s(x_train)
      d_w = self.derivatives(x_train,y_train)
      self.update(d_w)

  def predict(self, x_test):
    prob = self.s(x_test)
    return (prob >= np.median(prob)).astype(int)

def run_LR(x_train, x_test, y_train, y_test, lam):
    lr = LR(alpha=0.00001, lam=lam)
    lr.fit(x_train, y_train, epochs=5000)
    y_pred = lr.predict(x_test)
    report = classification_report(y_test, y_pred, target_names=["Negative", "Positive"], output_dict=True)
    f1 = report['weighted avg']['f1-score']
    return f1

