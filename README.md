# 📈 Stock Market Price Predictor

This project is a **Streamlit web application** that predicts stock prices using a **deep learning LSTM model** trained on historical stock data.

## 🚀 Features

- Predicts stock prices using a stacked LSTM model
- Fetches historical data using the `yfinance` API
- Visualizes:
  - Moving Averages (MA50, MA100, MA200)
  - Actual vs Predicted Prices
  - Next 30 Days Forecast
- Tabular view of predicted future prices
- Dynamic stock symbol input

---

## 📊 Algorithms Used

- **LSTM (Long Short-Term Memory)** neural networks for time series forecasting
- **MinMaxScaler** for normalization of stock prices
- Recursive prediction loop for multi-step future forecasting

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- scikit-learn
- Streamlit
- matplotlib
- yfinance
- pandas, numpy

---

## 📦 How to Run

### 1. Clone the repo
git clone https://github.com/your-username/stock-market-predictor.git
cd stock-market-predictor

### 2. Set up a virtual environment
py -3.11 -m venv .venv
.venv\Scripts\activate   # On Windows
source .venv/bin/activate  # On Linux/Mac


### 3. Install dependencies
pip install -r requirements.txt


### 4. Run the app
streamlit run app.py


## 🧠 Training Details

* Trained on 10 years of data (2015–2025)
* 4 LSTM layers with increasing neurons (50 → 120)
* Dropout applied to reduce overfitting
* Loss function: Mean Squared Error (MSE)

---

## 📈 Output

* Graphs comparing actual vs predicted prices
* 30-day future forecast using recursive prediction
* Tabular display of future predicted prices

---

## 📌 Note

This project is for educational purposes only and **should not be used for financial trading decisions**.

---

## 🧑‍💻 Author

Dea Mishra
Feel free to connect and contribute!
