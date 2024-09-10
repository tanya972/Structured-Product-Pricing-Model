import numpy as np
import yfinance as yf   
import matplotlib.pyplot as plt
from ticker import ticker
from iv import predicted_iv

def structured_product_pricing(S0, CouponRate, CouponBarrierPercent, KnockInBarrierPercent, AutocallBarrierPercent, Maturity, CouponDates, r, sigma, N):
    # Convert percentage barriers to absolute prices
    CouponBarrier = S0 * (CouponBarrierPercent / 100)
    KnockInBarrier = S0 * (KnockInBarrierPercent / 100)
    AutocallBarrier = S0 * (AutocallBarrierPercent / 100)
    
    dt = Maturity / N  # Length of each time step
    u = np.exp(sigma * np.sqrt(dt))  # Up factor
    d = 1 / u  # Down factor
    q = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability of up move

    # Initialize underlying asset prices at maturity
    ST = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            ST[j, i] = S0 * (u ** (i - j)) * (d ** j)

    # Initialize option values and coupon values
    option_values = np.zeros((N + 1, N + 1))
    coupon_values = np.zeros((N + 1, N + 1))

    # Set coupon values at coupon observation dates
    accrued_coupons = np.zeros((N + 1, N + 1))  # To store total accrued coupons at each step
    for i in CouponDates:
        for j in range(i + 1):
            if ST[j, i] >= CouponBarrier:
                coupon_values[j, i] = CouponRate / 100 * S0
            else:
                coupon_values[j, i] = 0

    # Backward induction with autocall feature and cumulative coupons
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            # Accumulate the coupon from previous steps
            accrued_coupons[j, i] = accrued_coupons[j, i + 1] + coupon_values[j, i]
            
            # If autocall condition is met at a coupon observation date, redeem early
            if i in CouponDates and ST[j, i] >= AutocallBarrier:
                print("AUTOCALLED:", ST[j,i])
                option_values[j, i] = S0 + accrued_coupons[j, i]  # Pay back initial investment plus accrued coupons
            else:
                exercise_value = accrued_coupons[j, i] + option_values[j, i]
                hold_value = np.exp(-r * dt) * (q * option_values[j, i + 1] + (1 - q) * option_values[j + 1, i + 1])
                option_values[j, i] = max(exercise_value, hold_value)

    # Final payoff at maturity
    for j in range(N + 1):
        accrued_coupons[j, N] += coupon_values[j, N]  # Include final coupon if paid
        if ST[j, N] >= KnockInBarrier:
            option_values[j, N] = S0 + accrued_coupons[j, N]  # Full principal repayment if above Knock-In barrier
        else:
            option_values[j, N] = ST[j, N]  # Loss if below Knock-In barrier

    # Intitalize timesteps for plot
    time_steps = np.linspace(0, Maturity, N + 1)

    # Calculate the average payoff at each time step
    average_payoff = np.zeros(N + 1)
    for i in range(N + 1):
        average_payoff[i] = np.mean(option_values[:i + 1, i])
    
    # Plot actual underlying asset price
    underlying = yf.download(ticker, start="2022-03-11", end='2024-09-09')
    underlying_close_prices = underlying['Close'].values
    dates = underlying.index

    start_date = dates[0]
    dates_in_years = (dates - start_date).days / 365.25  # Convert days to years

    # Fit to timesteps
    underlying_interpolated = np.interp(time_steps, dates_in_years, underlying_close_prices)

    # Plot the structured product's payoff over time, actual underlying asset price and 
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, average_payoff, label='Average Structured Product Payoff')
    plt.plot(time_steps, underlying_interpolated, label='Actual Underlying Asset Price')
    plt.xlabel('Time (Years)')
    plt.ylabel('Payoff ($)')
    plt.title(f'{ticker} {Maturity} Yr {CouponRate}% Autocall Structured Product Payoff Over Time')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Print the whole table of option values
    print("Option Values Table:")
    for i in range(N + 1):
        print(f"Time Step {i}: {option_values[:i+1, i]}")

    # Return final price of the structured product
    return option_values[0, 0] # Final payoff value at the root

# Example parameters with percentage barriers
underlying = yf.download(ticker, start="2022-03-11", end='2024-09-09')
current_price = underlying.loc['2022-03-11', 'Close']

S0 = current_price  # Starting price
CouponRate = 6  # Assume 5% coupon
CouponBarrierPercent = 50  # % of underlying initial level
KnockInBarrierPercent = 70  # % of initial price
AutocallBarrierPercent = 70  # % of initial price for autocall
Maturity = 2.5 # aka 30 months
CouponDates = [5, 11, 17, 23]  # Corresponding to months 6, 12, 18, and 24
r = 0.037  # Risk-free rate
sigma = predicted_iv   # Example implied volatility
N = 30  # 30 months

# Price structured product and get option values table
price = structured_product_pricing(S0, CouponRate, CouponBarrierPercent, KnockInBarrierPercent, AutocallBarrierPercent, Maturity, CouponDates, r, sigma, N)
print(f"\nThe price of the structured product with an initial underlying asset price of ${S0:.2f} is: ${price:.2f}")
