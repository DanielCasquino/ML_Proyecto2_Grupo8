import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from svm import SVM
from knn import KNN
from logireg import LR
from dtree import DT
from bootstrap import *

np.random.seed(42)
path = './tos/cleaned_data/'

def encode(path):
  positives = []
  positives_directory = os.listdir(path + "PositiveRepeated")
  print("Loading positives...")
  for f in positives_directory:
    y, sr = librosa.load(path + "PositiveRepeated/" + f)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    fv = mfccs.mean(axis = 1)
    positives.append(fv)
  positives_array = np.array(positives)
  positives_array = np.insert(positives_array, 0, 1, axis=1)
  print("Loading finished!")

  negatives = []
  negatives_directory = os.listdir(path + "Negative")
  print("Loading negatives...")
  for f in negatives_directory:
    y, sr = librosa.load(path + "Negative/" + f)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    fv = mfccs.mean(axis = 1)
    negatives.append(fv)
  negatives_array = np.array(negatives)
  negatives_array = np.insert(negatives_array, 0, -1, axis=1)
  print("Loading finished!")

  print("Joining and shuffling loaded data...")
  result_array = np.concatenate((positives_array, negatives_array), axis=0)
  np.random.shuffle(result_array)
  y = result_array[:,0]
  x = result_array[:, 1:]
  print("Done!")
  return x, y

def matriz_confusion(y_pred, y_test, Tipo):
  matrix = confusion_matrix(y_test, y_pred)
  f2 = pd.DataFrame(matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis], index=["Negative", 'Positive'], columns=["Negative", 'Positive'])
  sns.heatmap(f2, annot=True, cbar=None, cmap="Greens")
  plt.title("Confusion Matrix"  + Tipo ), plt.tight_layout()
  plt.xlabel("Predicted")
  plt.ylabel("Real")
  plt.show()

X, Y = encode(path)
sm = SMOTE(random_state=42)
X, Y = sm.fit_resample(X, Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

def test_svm():
  print("Training SVM...")
  model = SVM(1e8, 1e-10, 39000)
  model.train(x_train, y_train)
  y_pred = model.predict(x_test)
  print(y_pred)
  print(y_test)
  matriz_confusion(y_pred, y_test, ": SVM")

  report = classification_report(y_test, y_pred, target_names = ["Negative", "Positive"])
  print("Metrics")
  print(report)


def test_knn():
  print("Training KNN...")
  model = KNN()
  model.fit(x_train, y_train)
  y_pred = model.predict(x_test)
  print(y_pred)
  print(y_test)
  matriz_confusion(y_pred, y_test, ": KNN")

  report = classification_report(y_test, y_pred, target_names = ["Negative", "Positive"])
  print("Metrics")
  print(report)

def test_lr():
  model = LR(1e-2, 1e6)
  model.fit(x_train, y_train, 5000)
  y_pred = model.predict(x_test)
  print(y_pred)
  print(y_test)
  matriz_confusion(y_pred, y_test, ": LR")

  report = classification_report(y_test, y_pred, target_names = ["Negative", "Positive"])
  print("Metrics")
  print(report)


def test_dt():
  model = DT(x_train, y_train)
  y_pred = model.predict(x_test)
  print(y_pred)
  print(y_test)
  matriz_confusion(y_pred, y_test, ": DT")

  report = classification_report(y_test, y_pred, target_names = ["Negative", "Positive"])
  print("Metrics")
  print(report)

### LLAMA ESTA FUNCION PARA PROBAR EL MODELO
def exec_knn():
    k = [i for i in range(1,11)]
    f1_scores_knn = []
    for i in k:
        f1 = run_KNN(x_train, x_test, y_train, y_test, i)
        f1_scores_knn.append(f1)

    plt.plot(k, f1_scores_knn)
    plt.title("Evolucion de f1 scores del KNN")
    plt.xlabel("Valor de k_neighbors")
    plt.ylabel("F1-Score")
    plt.grid()
    plt.show()

    KNN_Kfold(X, Y)
    test_knn_bootstrapped(X, Y)

    knn = NearestNeighbor(k=2)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    report = classification_report(y_test, y_pred, target_names=["Negative", "Positive"])
    print(report)
    matriz_confusion(y_pred, y_test, " : LR")

### LLAMA ESTA FUNCION PARA PROBAR EL MODELO
def exec_lr():
    k = [10**i for i in range(-5,1)]
    f1_scores_lr = []
    for i in k:
        f1 = run_LR(x_train, x_test, y_train, y_test, i)
        f1_scores_lr.append(f1)

    plt.plot(k, f1_scores_lr)
    plt.title("Evolucion de f1 scores del LR")
    plt.xlabel("Valor del lambda")
    plt.ylabel("F1-Score")
    plt.grid()
    plt.show()

    LR_Kfold(X, Y)
    test_lr_bootstrapped(X, Y)

    lr = LR(alpha=0.00001, lam=0.01)
    lr.fit(x_train, y_train, epochs=5000)
    y_pred = lr.predict(x_test)
    report = classification_report(y_test, y_pred, target_names=["Negative", "Positive"])
    print(report)
    matriz_confusion(y_pred, y_test, " : LR")

#exec_lr() ---> para ver la regresion logistica
#exec_knn() ---> para ver el KNN


test_knn_bootstrapped(x_train, y_train)
