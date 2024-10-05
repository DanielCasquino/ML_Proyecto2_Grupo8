import numpy as np
import librosa
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
from knn import NearestNeighbor, get_best_k

os.environ['LOKY_MAX_CPU_COUNT'] = '4'

path = './cleaned_data/'

def encode(path, type1 = "Positive", type2 = "Negative"):
    data1 = []
    directory_1 = os.listdir(path + type1)
    for f  in directory_1:
        y, sr = librosa.load(path +  type1 + "/" + f)
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        fv = mfccs.mean(axis = 1)
        data1.append(fv)
    data = np.array(data1)
    data = np.insert(data, 0, 1, axis=1)

    data2 = []
    directory_2 =  os.listdir(path + type2)
    for f  in directory_2:
        y, sr = librosa.load(path + type2 + "/" + f)
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        fv = mfccs.mean(axis = 1)
        data2.append(fv)
    temp = np.array(data2)
    temp = np.insert(temp, 0, -1, axis=1)

    data = np.concatenate((data, temp), axis=0)
    np.random.shuffle(data)
    y = data[:,0]
    x = data[:, 1:]
    return x, y

X, Y = encode(path)
sm = SMOTE(random_state=42)
X, Y = sm.fit_resample(X, Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

def matriz_confusion(y_pred, y_test, Tipo):
    matrix = confusion_matrix(y_test, y_pred)
    f2 = pd.DataFrame(matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis], index=["Negative", "Positive"], columns=["Negative", "Positive"])
    sns.heatmap(f2, annot=True, cbar=None, cmap="Greens")
    plt.title("Confusion Matrix"  + Tipo ), plt.tight_layout()
    plt.xlabel("Predicted")
    plt.ylabel("Real")
    plt.show()

# Train with K-fold
k_fold = get_best_k(X, Y, 10)
print(k_fold)

# Testing with own implementation
knn = NearestNeighbor(2)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
matriz_confusion(y_pred, y_test, " : KNN")
