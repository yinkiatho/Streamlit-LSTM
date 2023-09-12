from __future__ import division
from keras import callbacks
import pandas as pd
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten, MaxPooling1D, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("RNN Trading/Datasets/SPY.csv", index_col="Date", parse_dates=True)


data.info()
data.drop(['Dividends', 'Stock Splits', "Capital Gains"], axis=1, inplace=True)


#Split train, test
scaler = MinMaxScaler(feature_range=(0, 1))
data = pd.DataFrame(scaler.fit_transform(data))
train, test = data.iloc[:int(len(data) * 0.7),:], data.iloc[int(len(data) * 0.7):len(data), :]


# Normalizing the data
def create_window(data, window_size=1):
    data_s = data.copy()
    for i in range(window_size):
        data = pd.concat([data, data_s.shift(-(i + 1))], axis=1)

    data.dropna(axis=0, inplace=True)
    return (data)


def reconstruct_data(data, n=1):
    x, y = [], []
    for i in range(len(data) - n - 1):
        x.append(data.iloc[i:(i + n), :])
        # append close price
        y.append(data.iloc[i + n, 3])

    return (np.array(x), np.array(y))


train_x, train_y = reconstruct_data(train, 5)
test_x, test_y = reconstruct_data(test, 5)

# print(train_x.shape, train_y.shape)


print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)


# Changing it into rnn input shape ([samples, time steps, features])
train_x = np.reshape(
    train_x, (train_x.shape[0], train_x.shape[1], train_x.shape[2]))
test_x = np.reshape(
    test_x, (test_x.shape[0], test_x.shape[1], test_x.shape[2]))

# train_x = np.reshape(
#    train_x, (train_x.shape[0], 1, 5* 18))
# test_x = np.reshape(
#    test_x, (test_x.shape[0], 1, 5 * 18))
print(train_x.shape)
print(test_x.shape)


def model_lstm(window, features):

    model4 = Sequential()
    model4.add(LSTM(300, input_shape=(window, features), return_sequences=True))
    model4.add(Dropout(0.5))
    model4.add(LSTM(200, input_shape=(window, features), return_sequences=True))
    model4.add(Dropout(0.5))
    model4.add(LSTM(100, input_shape=(window, features), return_sequences=False))
    model4.add(Dropout(0.5))
    model4.add(Dense(1))
    model4.compile(loss='mse', optimizer='adam')

    return model4

    return model

def model_lstms(window, features):
    
    model1 = Sequential()
    model1.add(LSTM(300, input_shape=(window, features), return_sequences=True))
    model1.add(Dropout(0.5))
    model1.add(LSTM(200, input_shape=(window, features), return_sequences=True))
    model1.add(Dropout(0.5))
    model1.add(LSTM(100, input_shape=(window, features), return_sequences=False))
    model1.add(Dropout(0.5))
    model1.add(Dense(1))
    model1.compile(loss='mse', optimizer='adam')

    model2 = Sequential()
    model2.add(LSTM(300, input_shape=(window, features), return_sequences=True))
    model2.add(Dropout(0.5))
    model2.add(LSTM(200, input_shape=(window, features), return_sequences=False))
    model2.add(Dropout(0.5))
    model2.add(Dense(100))
    model2.add(Dense(1))
    model2.compile(loss='mse', optimizer='adam')
    
    model3 = Sequential()
    model3.add(LSTM(300, input_shape=(window, features), return_sequences=True))
    model3.add(Dropout(0.5))
    model3.add(LSTM(200, input_shape=(window, features), return_sequences=True))
    model3.add(Dropout(0.5))
    model3.add(LSTM(100, input_shape=(window, features), return_sequences=False))
    model3.add(Dropout(0.5))
    model3.add(Dense(100))
    model3.add(Dense(1))
    model3.compile(loss='mse', optimizer='adam')


    return (model1, model2, model3)



earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                        mode="min", patience=10,
                                        restore_best_weights=True)


#Model Loss vs Epoch
models = model_lstm(5, 18)
curr_test_loss = 1000000
best_model = None
i = 1

for model in [models]:
    
    history = model.fit(train_x, train_y, epochs=500, 
                    batch_size=24, validation_data=(test_x, test_y),
                    verbose=0, callbacks=[], shuffle=False)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()


    #Evaluation
    test_loss = model.evaluate(test_x, test_y)
    print(f"Test Loss Model {i}:", test_loss)
    i += 1
    if test_loss < curr_test_loss:
        curr_test_loss = test_loss
        best_model = model


#Prediction on training set
y_predicted=best_model.predict(train_x)

#unscaled_predicted = scaler.inverse_transform(y_predicted)

plt.figure(figsize=(30, 10))
plt.plot(train_y, label="actual")
plt.plot(y_predicted, label="prediction")
plt.legend(fontsize=20)
plt.grid(axis="both")
plt.title("Actual close price and pedicted one on train set", fontsize=25)
plt.show()



#Predicting Testing Dataset
test_predict = best_model.predict(test_x)
#test_predict = scaler.inverse_transform(test_predict)
#test_labels = scaler.inverse_transform([test_y])

plt.figure(figsize=(30, 10))
plt.plot(test_y, label="actual")
plt.plot(test_predict, label="prediction")
plt.legend(fontsize=20)
plt.grid(axis="both")
plt.title("Actual close price and pedicted one on test set", fontsize=25)
plt.show()
