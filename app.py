import numpy as np # type: ignore
import pandas as pd # type: ignore
import yfinance as yf  # type: ignore
from keras.models import load_model  # type: ignore
import streamlit as st  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

model = load_model("Stock Predictions Model.keras")

st.header('Stock Market Price Predictor System')

stock =st.text_input('Enter Stock Symnbol', 'GOOG')
start = '2015-01-01'
end = '2025-05-10'

data = yf.download(stock, start ,end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler  # type: ignore
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')

ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))

plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(data.Close, 'g', label='Closing Price')

plt.legend(loc='upper left')  # or try 'best'
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)

st.pyplot(fig1)


st.subheader('Price vs MA50 vs MA100')

ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))

plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(ma_100_days, 'b', label='MA100')
plt.plot(data.Close, 'g', label='Closing Price')

plt.legend(loc='best')  # You can also use 'upper left', etc.
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)

st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')

ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))

plt.plot(ma_100_days, 'r', label='MA100')
plt.plot(ma_200_days, 'b', label='MA200')
plt.plot(data.Close, 'g', label='Closing Price')

plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)

st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)


future_input = data_test_scale[-100:]  # Last 100 values
future_input = future_input.reshape(1, future_input.shape[0], 1)

future_pred = []

for _ in range(30):
    next_pred = model.predict(future_input)[0][0]
    future_pred.append(next_pred)
    
    # Update input window with predicted value
    next_input = np.append(future_input[0, 1:], [[next_pred]], axis=0)
    future_input = next_input.reshape(1, 100, 1)

# Rescale predictions to original scale
future_pred_actual = np.array(future_pred) * scale

# Create dates for future predictions
last_date = pd.to_datetime(data.index[-1])
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 31)]


# Show Predicted Future Prices in Table Format
st.subheader('Future Prices Table (Next 30 Days)')

future_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Price': future_pred_actual.flatten()
})
st.dataframe(future_df)

fig5 = plt.figure(figsize=(8,6))
plt.plot(future_dates, future_pred_actual, color='blue', label='Forecasted Price')
plt.title('Next 30 Days Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
st.pyplot(fig5)