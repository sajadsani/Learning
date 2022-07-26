from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
# import tenserflow and keras
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

#print(np.shape(X_train_full),np.shape(X_train),np.shape(X_valid),np.shape(y_train))

if 1==0: # building sequential keras model mse:0.35
    model = keras.models.Sequential([keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),keras.layers.Dense(1)])
    model.compile(loss="mean_squared_error", optimizer="sgd")
    history = model.fit(X_train, y_train, epochs=20,validation_data=(X_valid, y_valid))
    mse_test = model.evaluate(X_test, y_test)
    print(mse_test)
    X_new = X_test[:3] # pretend these are new instances
    y_pred = model.predict(X_new)
elif 1==0: # building a functional API with wide and deep architecture  mse=0.38
    # Building the model
    input_ = keras.layers.Input(shape=X_train.shape[1:])
    hidden1 = keras.layers.Dense(30, activation="relu")(input_)
    hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
    concat = keras.layers.concatenate([input_, hidden2])
    output = keras.layers.Dense(1)(concat)
    model = keras.Model(inputs=[input_], outputs=[output])
    # compiling and training the model
    model.compile(loss="mean_squared_error", optimizer="sgd")
    history = model.fit(X_train, y_train, epochs=20,validation_data=(X_valid, y_valid))
    mse_test = model.evaluate(X_test, y_test)
    print(mse_test)
    X_new = X_test[:3] # pretend these are new instances
    y_pred = model.predict(X_new)
elif 1==0: # building a multiple input functional API with keras   mse = 0.67
    input_A = keras.layers.Input(shape=[5], name="wide_input")
    input_B = keras.layers.Input(shape=[6], name = "deep_input")
    hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
    hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
    concat = keras.layers.concatenate([input_A, hidden2])
    output = keras.layers.Dense(1, name="output")(concat)
    model = keras.models.Model(inputs=[input_A, input_B], outputs=[output])
    # compiling and traini the model
    model.compile(loss="mse",optimizer=keras.optimizers.SGD(lr=1e-3))
    # to define new splitted input
    X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
    X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
    X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
    X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]
    # to train model
    history = model.fit((X_train_A,X_train_B),y_train, epochs=20, validation_data=((X_valid_A,X_valid_B),y_valid))
    mse_test = model.evaluate((X_test_A,X_test_B),y_test)
    print(mse_test)
    y_pred = model.predict((X_new_A,X_new_B))
elif 1==1:
    input_A = keras.layers.Input(shape=[5], name="wide_input")
    input_B = keras.layers.Input(shape=[6], name = "deep_input")
    hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
    hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
    concat = keras.layers.concatenate([input_A, hidden2])
    output = keras.layers.Dense(1, name="output")(concat)
    aux_output = keras.layers.Dense(1, name="aux_output")(hidden2)
    model = keras.models.Model(inputs=[input_A, input_B], outputs=[output,aux_output])
    # to compile the model
    model.compile(loss = ["mse","mse"], loss_weights = [0.9,0.1],optimizer = "sgd")
    # to define new splitted input
    X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
    X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
    X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
    X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]
    # to train the model
    if 1==0: # to train model and save it at the end of training
        history = model.fit([X_train_A,X_train_B],[y_train,y_train], epochs=20, validation_data=([X_valid_A,X_valid_B],[y_valid,y_valid]))
        total_loss, main_loss, aux_loss = model.evaluate([X_test_A, X_test_B], [y_test, y_test])
        y_pred_main, y_pred_aux = model.predict([X_new_A, X_new_B])
        model.save("my_keras_model.h5")
        # if we want to load a model
        model = keras.models.load_model("my_keras_model.h5")
    elif 1==1: # to use callback to save model at each apoch and at early stopping
        # checkpoint callback:
        checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5",save_best_only=True)
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)
        history = model.fit([X_train_A,X_train_B],[y_train,y_train], epochs=20, validation_data=([X_valid_A,X_valid_B],[y_valid,y_valid]),callbacks=[checkpoint_cb, early_stopping_cb])
        total_loss, main_loss, aux_loss = model.evaluate([X_test_A, X_test_B], [y_test, y_test])
        y_pred_main, y_pred_aux = model.predict([X_new_A, X_new_B])



