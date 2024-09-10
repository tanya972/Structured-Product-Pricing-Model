import numpy as np
import pandas as pd
import yfinance as yf   
import statsmodels.api as sm
from datetime import datetime
from ticker import ticker

today_date = datetime.today().strftime('%Y-%m-%d')
# Download historical data for Bitcoin
data = yf.download(ticker, start="2020-01-01", end=today_date)

# Calculate the log returns
data['log_returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
data = data.dropna()

# Annualized historical volatility for Bitcoin
historical_volatility = data['log_returns'].std() * np.sqrt(252)

# Sample implied volatilities for different maturities (synthetic data)
np.random.seed(42) 
maturities = np.linspace(0.5, 5, 100)  # Maturities from 0.5 to 5 years
historical_volatilities = np.random.normal(historical_volatility, 0.1, size=len(maturities)) #0.1 standard dev

# Assume implied volatility is slightly higher than historical and depends on maturity (0.05 slope). Random noise is 0.02 std dev
implied_volatilities = historical_volatilities + 0.05 * maturities + np.random.normal(0, 0.02, size=len(maturities))

# Create a DataFrame for regression
df = pd.DataFrame({
    'Implied_Volatility': implied_volatilities,
    'Historical_Volatility': historical_volatilities,
    'Maturity': maturities
})

# Linear Regression Model
X = df[['Historical_Volatility', 'Maturity']]
X = sm.add_constant(X) # add intercept 1 

Y = df['Implied_Volatility']

#fit model
model = sm.OLS(Y,X).fit()
#print(model.summary())
rsqr = model.rsquared
print(f"rsquared:{rsqr:.2f}")

# Predict implied volatility 
maturity_desired = 2.5
historical_volatility_for_prediction = historical_volatility 

predicted_iv = model.predict([1, historical_volatility_for_prediction, maturity_desired])
print(f"Estimated Implied Volatility for {ticker} over 2.5 years: {predicted_iv[0]:.2%}")