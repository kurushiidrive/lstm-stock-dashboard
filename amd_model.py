#!/usr/bin/env python
# coding: utf-8

#


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense

scaler = MinMaxScaler(feature_range=(0,1))
SPLIT = 6732

# obtain dataframe from csv file
df = pd.read_csv('https://raw.githubusercontent.com/kurushiidrive/lstm-stock-dashboard/master/datasets_541298_1054465_stocks_AMD.csv')
df.tail()


#


df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
df.index = df['Date']

plt.figure(figsize=(16,8))
plt.plot(df['Close'], label='Close Price History')
plt.xlabel('Year')
plt.ylabel('Closing Price (USD)')
plt.title('AMD Closing Price from 17 Mar 1980 to 1 April 2020')
plt.show()


#


# sort data from earliest date to latest
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date','Close'])

# copy sorted data into a new dataframe containing only the 'Date' and 'Close' columns
for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

# normalise the new dataframe
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

final_data = new_data.values                 # a single column containing all of the closing prices from new_data

# fragmentations of final_data
train_data = final_data[0:SPLIT, :]               # just a single column containing a row for each closing price for the first X days
valid_data = final_data[SPLIT:, :]                # just a single column containing a row for each closing price for the remaining len(new_data)-X days

scaled_data = scaler.fit_transform(final_data)  # normalisation

# place normalised data into X_train and y_train
X_train, y_train = [], []

# use the last 60 closing prices to predict the next closing price
for i in range(60, len(train_data)):
    X_train.append(scaled_data[i-60:i-1, 0])
    y_train.append(scaled_data[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


#


# DEBUGGING
print('final_data')
print(str(final_data) + '\n')

print('new_data')
print(str(new_data) + '\n')

print('train_data')
print(str(train_data) + '\n')

print('valid_data')
print(str(valid_data) + '\n')

print('X_train')
print(X_train.shape)
print(str(X_train) + '\n')

print('y_train')
print(y_train.shape)
print(str(y_train) + '\n')


#


# build keras Sequential model with LSTM (long short-term memory) and Dense layers
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=2)
model.summary()


#


# input_data will contain normalised validation data which the trained model will make predictions on.
input_data = new_data[len(new_data)-len(valid_data)-60:].values
input_data = input_data.reshape(-1,1)
input_data = scaler.transform(input_data)

# X_test is merely a reshaped copy of input_data
X_test = []
for i in range(60, input_data.shape[0]):
    X_test.append(input_data[i-60:i-1, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_closing_price = model.predict(X_test)
predicted_closing_price = scaler.inverse_transform(predicted_closing_price) # predictions made on the input_data are now stored in predicted_closing_price, and unnormalised.


#


model.save("amd_model")


#


train_data = new_data[:SPLIT]
valid_data = new_data[SPLIT:]
valid_data['Predictions'] = predicted_closing_price
plt.plot(train_data['Close'], label='Training Data')
plt.plot(valid_data['Close'], label='Validation Data')
plt.plot(valid_data['Predictions'], label='Predictions', color='green')
plt.xlabel('Year')
plt.ylabel('Closing Price (USD)')
plt.legend(loc='upper left')
plt.show()


#


plt.plot(train_data['Close'], label='Training Data')
plt.plot(valid_data['Close'], label='Validation Data', color='orange')
plt.xlabel('Year')
plt.ylabel('Closing Price (USD)')
plt.legend(loc='upper left')
plt.show()


#


# DEBUGGING
print('final_data')
print(str(final_data) + '\n')

print('new_data')
print(str(new_data) + '\n')

print('train_data')
print(str(train_data) + '\n')

print('valid_data')
print(str(valid_data) + '\n')

print('input_data')
print(str(input_data) + '\n')

print('input_data_prenormalise')
print(str(new_data[len(new_data)-len(valid_data)-60:].values) + '\n')

