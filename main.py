import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from svm import SVM
from knn import KNN

path = './cleaned_data/'

def encode(path):
  positives = []
  positives_directory = os.listdir(path + "Positive")
  print("Loading positives...")
  for f in positives_directory:
    y, sr = librosa.load(path + "Positive/" + f)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    fv = mfccs.mean(axis = 1)
    positives.append(fv)
  positives_array = np.array(positives)
  positives_array = np.insert(positives_array, 0, 1, axis=1)
  print("Loading finished!")

  negatives = []
  negatives_directory = os.listdir(path + "test_neg")
  print("Loading negatives...")
  for f in negatives_directory:
    y, sr = librosa.load(path + "test_neg/" + f)
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

X_data, Y_data = encode(path)
x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3)

model = SVM(1e8, 1e-10, 6000)
model.train(x_train, y_train)
y_pred = model.predict(x_test)
print(y_pred)
print(y_test)
matriz_confusion(y_pred, y_test, " : SVM")

report = classification_report(y_test, y_pred, target_names = ["Negative", "Positive"])
print("Metrics")
print(report)
