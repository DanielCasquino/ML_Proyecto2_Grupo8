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

import numpy as np

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
