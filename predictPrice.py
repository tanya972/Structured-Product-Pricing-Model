import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Fetch historical data
ticker = 'AAPL'  # Replace with your stock ticker symbol
data = yf.download(ticker, start='2020-01-01', end='2023-01-01')

# Feature Engineering: Add moving averages
data['MA50'] = data['Close'].rolling(window=50).mean()  # 50-day moving average
data['MA200'] = data['Close'].rolling(window=200).mean()  # 200-day moving average
data['Volume'] = data['Volume']  # Volume feature

# Drop any rows with NaN values (due to moving averages)
data = data.dropna()

# Prepare features and target
data['Days'] = (data.index - data.index.min()).days  # Days since start
X = data[['Days', 'MA50', 'MA200', 'Volume']]  # Multiple features
y = data['Close']  # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model.fit(X_train, y_train)

# Predict on test data
y_pred = rf_model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Predict for the next 2.5 years
last_date = data.index.max()
future_dates = pd.date_range(last_date, periods=int(2.5 * 365), freq='D')
future_days = (future_dates - data.index.min()).days
future_ma50 = np.full(len(future_days), data['MA50'].iloc[-1])  # Extend the last known MA50
future_ma200 = np.full(len(future_days), data['MA200'].iloc[-1])  # Extend the last known MA200
future_volume = np.full(len(future_days), data['Volume'].iloc[-1])  # Extend the last known volume

# Prepare future feature set
future_X = pd.DataFrame({
    'Days': future_days,
    'MA50': future_ma50,
    'MA200': future_ma200,
    'Volume': future_volume
})

# Predict future stock prices
future_prices = rf_model.predict(future_X)

# Create a DataFrame for future predictions
future_data = pd.DataFrame({'Date': future_dates, 'Predicted_Price': future_prices})
future_data.set_index('Date', inplace=True)

# Plot historical and future prices
plt.figure(figsize=(12, 6))

# Plot historical data
plt.plot(data.index, data['Close'], color='black', label='Historical Prices')

# Plot future predictions
plt.plot(future_data.index, future_data['Predicted_Price'], color='blue', label='Predicted Prices')

# Customize plot
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f'{ticker} Stock Price Prediction with Advanced Features')
plt.legend()
plt.show()