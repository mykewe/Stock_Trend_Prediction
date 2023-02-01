import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data
from keras.models import load_model
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import date
yf.pdr_override()

start = '2010-01-01'
today = date.today()
end = today.strftime("%Y-%m-%d")


st.title('Stock Trend Prdiction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = data.DataReader(user_input, start, end)

#Describe data
st.subheader('Data from 2010-Present')
st.write(df.describe())

#Visualizations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize =(12,6))
plt.plot(df.Close)
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100 days Moving Average (100MA)')
ma100= df.Close.rolling(100).mean()
fig = plt.figure(figsize =(12,6))
plt.plot(ma100, 'b', label='100MA')
plt.plot(df.Close, 'g', label='Close')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100 & 200 days Moving Average (100MA & 200MA)')
ma100= df.Close.rolling(100).mean()
ma200= df.Close.rolling(200).mean()
fig = plt.figure(figsize =(12,6))
plt.plot(ma100, 'b', label='100MA')
plt.plot(ma200, 'r', label='200MA')
plt.plot(df.Close, 'g', label='Close')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

# Splitting Data into Training and Testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

#Load model
model = load_model('keras_model.h5')

#Test data
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index = True)

#Scale data
scaler = MinMaxScaler(feature_range=(0,1))
input_data = scaler.fit_transform(final_df)

#Define features and targets
x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test, y_test = np.array(x_test), np.array(y_test)


#Making Prediction
y_pred = model.predict(x_test)

scale_factor = 1/scaler.scale_
y_pred*=scale_factor
y_test*=scale_factor


st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_pred, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

