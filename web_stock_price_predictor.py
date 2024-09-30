import streamlit as st
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import yfinance as yf
import matplotlib.pyplot as plt

model = load_model('stock_price_model.keras')

st.title("Stock Price Prediction App")

st.sidebar.title("Stock Settings")
stock_symbol = st.sidebar.text_input("Enter stock symbol", "GOOGL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2012-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

data = yf.download(stock_symbol, start=start_date, end=end_date)

st.subheader(f"Stock Data for {stock_symbol} from {start_date} to {end_date}")
st.write(data)

st.subheader('Close Price History')
plt.figure(figsize=(16, 8))
plt.plot(data['Close'])
plt.title(f'Close Price History of {stock_symbol}')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
st.pyplot(plt)

df = data.filter(['Close'])
dataset = df.values
training_data_len = math.ceil(len(dataset) * 0.8)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

train = df[:training_data_len]
valid = df[training_data_len:]
valid['Predictions'] = predictions

st.subheader('Model Predictions vs Actual')
plt.figure(figsize=(16, 8))
plt.plot(train['Close'], label='Train')
plt.plot(valid[['Close', 'Predictions']], label='Validation & Prediction')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.legend(['Train', 'Validation', 'Predictions'], loc='lower right')
st.pyplot(plt)

st.write(valid)

st.subheader(f"Predicting {stock_symbol}'s Price for Next Day")
last_60_days = df[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
st.write(f"Predicted price for the next day: ${pred_price[0][0]:.2f}")
