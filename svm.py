r"""
SupportVectorMachine
===

This module contains an implementation of a SVM class for our Machine Learning project.

This module exports:
  - SVM class

Usage
-----
To create an instance of the SVM class, do::

  >>> model = SVM(1e8, 1e-10, 4000)

Constructor parameters are (in order):
  - `c` the regularization constant
  - `alpha` the alpha constant - controls model reaction to slope change
  - `epochs` number of training iterations

"""

import numpy as np

class SVM:
  def __init__(self, c= 1e8, alpha = 1e-10, epochs = 39000):
    self.c = c
    self.alpha = alpha
    self.epochs = epochs
    self.w = None
    self.b = None
    self.error = []

  def normalize(self, data):
    min = np.min(data)
    max = np.max(data)
    return ((data - min)/(max - min))

  def h(self, x):
    return np.dot(x, self.w.T) + self.b

  def loss(self, y, y_aprox):
    return 1/2 * np.linalg.norm(self.w)**2 + self.c * np.sum(np.maximum(0, 1 - y * y_aprox))

  def derivatives(self, x, y, y_aprox):
      n = x.shape[0]
      dw = np.zeros(self.w.shape)
      db = 0.0

      for i in range(n):
          if (y[i]*y_aprox[i]) < 1:
              dw += self.c * -y[i] * x[i]
              db += self.c * -y[i]
      dw += self.w

      return dw, db

  def update(self, x, y, y_aprox, dw, db):
      n = x.shape[0]
      for i in range(n):
          if y[i] * y_aprox[i] < 1:
              self.w -= self.alpha * (dw - self.w)
              self.b -= self.alpha * db
          else:
              self.w -= self.alpha * self.w

  def train(self, x, y):
    x = self.normalize(x)
    self.w = np.array([np.random.rand() for _ in range(x.shape[1])])
    self.b = np.random.rand()

    for _ in range(self.epochs):
      y_aprox = self.h(x)
      dw, db = self.derivatives(x, y, y_aprox)
      self.update(x, y, y_aprox, dw, db)
      L = self.loss(y, y_aprox)
      self.error.append(L)
    return self.w, self.b, self.error

  def predict(self, x):
    x = self.normalize(x)
    y_aprox = []
    for i in range(x.shape[0]):
      y_aprox.append(np.sign(np.dot(x[i], self.w.transpose()) + self.b))
    return np.array(y_aprox)