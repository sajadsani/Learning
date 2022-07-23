import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron # scikit learn has a single TLU preceptron architecture
# import tenserflow and keras
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

if 1==0:# test installation of tenserflow and keras
    print(tf.__version__)
    print(keras.__version__)

if 1==0: # using scikit-learn perceptron for (single layer) for classification on iris dataset
    iris = load_iris()
    X = iris.data[:, (2, 3)] # petal length, petal width
    y = (iris.target == 0).astype(np.int) # Iris Setosa?
    per_clf = Perceptron()
    per_clf.fit(X, y)
    y_pred = per_clf.predict([[2, 0.5]])
    print(y_pred)

# using keras and tensorflow for classification on Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
if 1==0:
    print(X_train_full.shape)
    print(X_train_full.dtype)
# creating validation and train sets and also scaling data to 0-1 scale
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
# labeling class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat","Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# Creating a keras model with one input layer (transforming date to 1D array), and two hidden dense layer with relu activation function, and one output softmax layer
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
print(model.summary())
print(model.layers)
hidden1 = model.layers[1]
print(model.layers[1].name)
model.get_layer('dense') is hidden1
weights, biases = hidden1.get_weights()
print(weights)

# compiling the model
model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])
# training the model
history = model.fit(X_train, y_train, epochs=30,validation_data=(X_valid, y_valid))
# plot accuract and loss of main set and validation set
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()
# get the error estimation using evaluation on test set:
model.evaluate(X_test, y_test)
# predicting on nw instances:
X_new = X_test[:3] # some instances
y_proba = model.predict(X_new)
print(y_proba.round(2))
# or use predict class method
y_pred = model.predict_classes(X_new)
print(y_pred)
np.array(class_names)[y_pred]
