import numpy as np
import pandas as pd
# training path
training_data = pd.read_csv("Google_Stock_Price_Train.csv")

#load all and refer to col 1 ( 2nd actual col)
training_set = training_data.iloc[:, 1:2].values

# FEATURE SCALING standardization & normalization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_data_scaled = sc.fit_transform(training_set)

#create data struct with timesteps and 1 output
x_train = []
y_train = []

for i in range(60, 1258):
    x_train.append(training_data_scaled[i-60:i, 0])
    y_train.append(training_data_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Build the RNN and import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialize Add LTSM layer and provide some dropout (20%)
regressor = Sequential()

regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# 20 perc of he neurons will be ignored or rather 10 neurons in this case
regressor.add(Dropout(0.2))
#second LTSM layer

regressor.add(LSTM(units=50, return_sequences=True))
# 20 perc of he neurons will be ignored or rather 10 neurons in this case
regressor.add(Dropout(0.2))
# third LTSM layer

regressor.add(LSTM(units=50, return_sequences=True))
# 20 perc of he neurons will be ignored or rather 10 neurons in this case
regressor.add(Dropout(0.2))
# fourth LTSM layer

regressor.add(LSTM(units=50, return_sequences=False))
# 20 perc of he neurons will be ignored or rather 10 neurons in this case
regressor.add(Dropout(0.2))
#output layer
regressor.add(Dense(units=1))

#compile RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

#fitting the RNN to the training set

regressor.fit(x_train, y_train, epochs=100, batch_size=32)

