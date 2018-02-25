## Import
import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt

np.random.seed(42)

## Load data from keras (uses AWS)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)
print(x_train.shape)
print(x_test.shape)

## Examine data
print(x_train[0])
print(y_train[0])

## One-hot encoding the output into vector mode, each of length 1000
tokenizer = Tokenizer(num_words=1000)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print(x_train[0])

## One-hot encoding the output
num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train.shape)
print(y_test.shape)

## Building model structure
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=1000))
model.add(Dropout(.2))
model.add(Dense(num_classes, activation='softmax'))

## Compiling model using optimizer and loss function
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

## Training model with batch_size and epochs active
hist = model.fit(x_train, y_train, epochs=10, batch_size=20, validation_data = (x_test, y_test), verbose=2)

## Evaluating model
score = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: ", score[1])