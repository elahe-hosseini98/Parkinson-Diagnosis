import numpy as np
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random
from keras import backend as K
random.seed(10)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


size_of_files = []
line_of_files = []

#Iterate directory
dir_path = r'hw_dataset/parkinson/'
for file_name in os.listdir(dir_path):
    if os.path.isfile(os.path.join(dir_path, file_name)):
        line_of_file = []
        file = open(dir_path+file_name, 'r')
        lines = file.readlines()
        size_of_files.append(len(lines))
        lines = [line.split(";") for line in lines]
        for line in lines:
            line_of_file.append([line[0], line[1], line[3], line[4]])
        line_of_files.append(line_of_file)
 
        
dir_path = r'hw_dataset/control/'
for file_name in os.listdir(dir_path):
    if os.path.isfile(os.path.join(dir_path, file_name)):
        file = open(dir_path+file_name, 'r')
        lines = file.readlines()
        size_of_files.append(len(lines))
        lines = [line.split(";") for line in lines]
        for line in lines:
            line_of_file.append([line[0], line[1], line[3], line[4]])
        line_of_files.append(line_of_file)


#0, 1, 3, 4 => idx of features im going to use for classification
X = np.zeros((40, min(size_of_files), 4), dtype=np.uint8)
y = np.zeros((40, 1), dtype=np.uint8)
print(X.shape)

for i in range(40):
    X[i] = line_of_files[i][: min(size_of_files)]

for i in range(40):
    if i < 25:
        y[i] = 1
    else:
        y[i] = 0
    

X, y = shuffle(X, y, random_state=43)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size = .8)
print(X_train.shape)

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()
model.add(layers.Conv1D(filters=64, kernel_size=5, input_shape=(min(size_of_files), 4)))
model.add(layers.MaxPooling1D(pool_size=5 ))
model.add(layers.Conv1D(filters=32, kernel_size=5))
model.add(layers.MaxPooling1D(pool_size=5))
model.add(layers.Conv1D(filters=32, kernel_size=5))
model.add(layers.MaxPooling1D(pool_size=5))
model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
model.compile(loss='mse', optimizer='adam', metrics=['acc',f1_m,precision_m, recall_m])
model.fit(X_train, y_train, epochs=5)
loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test)
print('Test Loss: {}'.format(loss))
print('Test Accuracy: {}'.format(accuracy))
print('Test f1_score: {}'.format(f1_score))
print('Test precision: {}'.format(precision))
print('Test recall: {}'.format(recall))
print('='*70)
#==============================================================================

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam

model = Sequential()

#Adding the input LSTM network layer
model.add(LSTM(128, input_shape=(X.shape[1:]), return_sequences=True))

model.add(Dropout(0.2))
model.add(LSTM(128))

#Adding a dense hidden layer
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

#Adding the output layer
model.add(Dense(2, activation='softmax'))
model.compile( loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=0.001, decay=1e-6),
              metrics=['acc',f1_m,precision_m, recall_m])

#Fitting the data to the model
model.fit(X_train, y_train, epochs=5)

loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test)
print('Test Loss: {}'.format(loss))
print('Test Accuracy: {}'.format(accuracy))
print('Test f1_score: {}'.format(f1_score))
print('Test precision: {}'.format(precision))
print('Test recall: {}'.format(recall))




