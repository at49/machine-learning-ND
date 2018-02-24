import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

## import data
data = pd.read_csv("/Users/nubz/Desktop/python_ML/student_data.csv")
print(data)

## Visualize data
def plot_points(data):
    X = np.array(data[["gre","gpa"]])
    y = np.array(data["admit"])
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'red', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'cyan', edgecolor = 'k')
    plt.xlabel('Test (GRE)')
    plt.ylabel('Grades (GPA)')
plot_points(data)
plt.show()

## Visualize data by rank
data_rank1 = data[data["rank"]==1]
data_rank2 = data[data["rank"]==2]
data_rank3 = data[data["rank"]==3]
data_rank4 = data[data["rank"]==4]
plot_points(data_rank1)
plt.title("Rank 1")
plt.show()
plot_points(data_rank2)
plt.title("Rank 2")
plt.show()
plot_points(data_rank3)
plt.title("Rank 3")
plt.show()
plot_points(data_rank4)
plt.title("Rank 4")
plt.show()

## remove NaNs
data = data.fillna(0)

## One-hot encode the rank feature
processed_data = pd.get_dummies(data, columns=["rank"])

## Split data into X and y, and one-hot encode output as accepted/not accepted
## Normalize GRE and GPA scores in X prior to training model as values are 
## significantly different
import keras
from keras.utils import np_utils
from sklearn import preprocessing
X_raw = np.array(processed_data)[:, 1: ]
X_raw = X_raw.astype('float32')
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X_raw)
y = keras.utils.to_categorical(data["admit"],2)

## Check that input/output look as expected
print("Shape of X:", X.shape)
print("\nShape of y:", y.shape)
print("\nFirst 10 rows of X")
print(X[:10])
print("\nFirst 10 rows of y")
print(y[:10])

## Split data into training and testing sets
(X_train, X_test) = X[50:], X[:50]
(y_train, y_test) = y[50:], y[:50]

# Print shape of training set & no. of training & testing samples
print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

## Import models
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

## Build model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(7,)))
model.add(Dropout(.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(.1))
model.add(Dense(2, activation='softmax'))

## Compiling the model
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

## Train the model
model.fit(X_train, y_train, epochs=1000, batch_size=100, verbose=0)

## Evaluate and print model score
train_score = model.evaluate(X_train, y_train)
test_score = model.evaluate(X_test, y_test)
print("Training score: ", train_score[1])
print("Testing score: ", test_score[1])


