import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import svm

path = './tos/cleaned_data/'


def encode(path):
  positives = []
  positives_directory = os.listdir(path + "/Positive")
  print("Loading positives...")
  for f in positives_directory:
    y, sr = librosa.load(path + "/Positive/" + f)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    fv = mfccs.mean(axis = 1)
    positives.append(fv)
  positives_array = np.array(positives)
  positives_array = np.insert(positives_array, 0, 1, axis=1)
  print("Loading finished!")

  negatives = []
  negatives_directory = os.listdir(path + "/Negative")
  print("Loading negatives...")
  for f in negatives_directory:
    y, sr = librosa.load(path + "/Negative/" + f)
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


x_train, y_train = encode(path)
print(x_train.shape, y_train.shape)

