
import pandas as pd
import numpy as np
import statistics as st

from sklearn.model_selection import train_test_split

from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


import tensorflow as tf
import keras
from keras import backend as K
from sklearn.metrics import roc_auc_score
import time

from sklearn.model_selection import StratifiedKFold

from auxTensor import AuxTensor as at

import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")


all_releases_df = pd.read_csv('all_releases.csv')

#all_releases_df.head()

print("Generating a new dataframe without containing the last release...")
df = all_releases_df[all_releases_df['release'] != all_releases_df['release'].max()]
print("... DONE!")

df.drop(columns=['class_frequency', 'number_of_changes', 'release', 'Name', 'Kind', 'File'])


listResults = {} 
for i in [2, 3, 4, 5, 6]:
    print("window_size: "+str(i))
    listResults[i] = at.applyTensor(at,i, df)
    print("... DONE!")

for i in [2, 3, 4, 5, 6]:
    print("RESULTS FROM WINDOW SIZE",i)
    lastModel = listResults[i][0]
    history_general = listResults[i][1]
    X_test = listResults[i][2]
    y_test = listResults[i][3]
    time_in_seconds = listResults[i][4]

    fractions_t = 1-y_test.sum(axis=0)/len(y_test)
    weights_t = fractions_t[y_test.argmax(axis=1)]

    output = lastModel.predict_classes(X_test)
    print(confusion_matrix(y_test.argmax(axis=1), output))
    print(classification_report(y_test.argmax(axis=1), output))
    at.evaluation_metrics(y_test.argmax(axis=1), output, weights_t)
    print("Time in seconds:", time_in_seconds)
    
    print("\n")

