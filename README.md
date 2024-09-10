# Structured-Product-Pricing-Model

Models structured product pricing with Autocall, coupon and KnockInBarrier using Cox-Ross-Rubenstein model, inputting implied volatility derived from linear regression model.

# Sample Usage
Take ticker GBTC (Grayscale Bitcoin Trust). With parameters for the product as below, the average payoff at each timestep is shown below. We see it provides consistent payoff compared to the underlying asset.   

```python
CouponRate = 6  # Assume 5% coupon
CouponBarrierPercent = 50  # % of underlying initial level
KnockInBarrierPercent = 70  # % of initial price
AutocallBarrierPercent = 70  # % of initial price for autocall
Maturity = 2.5 # aka 30 months
CouponDates = [5, 11, 17, 23]  # Corresponding to months 6, 12, 18, and 24
r = 0.037  # Risk-free rate
sigma = 89.73 # Predicted Implied volatility of GBTC over 2.5 years  
```

![image](https://github.com/user-attachments/assets/00d8a9ff-786c-44b4-92d0-1be9e57b7868)

