import datetime as dt
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import yfinance as yf

from scipy.stats import norm

cwd = '/Users/vlr/PycharmProjects/thesis'
dl = '/Users/vlr/Downloads/Masters/python'

ticker = "TSLA"
dat = yf.Ticker(ticker)  # stock
expiry_dates = dat.options

# Get stock historical data
hist = pd.DataFrame(dat.history(period="5y"))  # Stock opening and closing prices for past 5 years
hist["GrossReturn"] = hist.Close.pct_change()  # Percentage change from one day to the other
hist["LogReturn"] = np.log(hist.Close) - np.log(hist.Close.shift(1))  # Logarithmic return
hist = hist.iloc[1:]
hist["Volatility"] = hist.LogReturn.rolling(window=20).std()*np.sqrt(252)
hist["sigma"] = hist.LogReturn.rolling(window=20).std()
# 20 daily returns = 1 trading month, annualized
hist = hist.dropna()

# Plot the stock price over time
plt.plot(hist["Close"])
plt.grid()
plt.ylabel(f'{ticker}' " stock price")
plt.show()

# Plot the volatility over time
plt.plot(hist['Volatility'])
plt.grid()
plt.ylabel(f'{ticker}' " volatility")
plt.show()

# Plot the risk-free rate
rf_plot = pd.read_csv(dl+"/10y_US_bond.csv")
rf_plot["Date"] = rf_plot["Date"].str.split("/")
for i in range(len(rf_plot)):
    rf_plot["Date"].iloc[i] = rf_plot["Date"][i][2] + rf_plot["Date"][i][0] + rf_plot["Date"][i][1]
rf_plot = rf_plot.sort_values(by='Date')
rf_plot = rf_plot.reset_index(drop=True)

plt.plot(rf_plot["Price"])
plt.ylabel("Risk free rate")
plt.grid()
plt.show()

'''
# Extract data for stock options: call
calls = {}
for i in range(len(expiry_dates)):
    key = str('call_'+str(expiry_dates[i]))
    calls[key] = pd.DataFrame(dat.option_chain(date=expiry_dates[i]).calls)
    calls[key]["expiryDate"] = str(expiry_dates[i])
    calls[key]["type"] = "call_option"

    end = dt.date(
        int(str.split(expiry_dates[i], "-")[0]),
        int(str.split(expiry_dates[i], "-")[1]),
        int(str.split(expiry_dates[i], "-")[2])
    )
    start = dt.date(2025, 4, 25)
    calls[key]["T"] = str(np.busday_count(start, end))
calls = pd.concat(calls.values(), ignore_index=True)

# Extract historical data for stock options: put
puts = {}
for i in range(len(expiry_dates)):
    key = str('call_'+str(expiry_dates[i]))
    puts[key] = pd.DataFrame(dat.option_chain(date=expiry_dates[i]).puts)
    puts[key]["expiryDate"] = str(expiry_dates[i])
    puts[key]["type"] = "put_option"

    end = dt.date(
        int(str.split(expiry_dates[i], "-")[0]),
        int(str.split(expiry_dates[i], "-")[1]),
        int(str.split(expiry_dates[i], "-")[2])
    )
    start = dt.date(2025, 4, 25)
    puts[key]["T"] = str(np.busday_count(start, end))
puts = pd.concat(puts.values(), ignore_index=True)
'''

# Combine all the date into one dataframe
hist["Date"] = hist.index.strftime('%m/%d/%Y')
hist = hist.reset_index(drop=True)
rf = pd.read_csv(dl+"/10y_US_bond.csv")
rf = rf.rename(columns={"Date": "Date", "Price": "rf"})
rf = rf.drop(['Open', 'High', 'Low', 'Change %'], axis=1)
data = pd.merge(hist, rf, how='left', on=['Date'])
data["Date"] = data["Date"].str.split("/")
for i in range(len(data)):
    data["Date"].iloc[i] = data["Date"][i][2] + data["Date"][i][0] + data["Date"][i][1]
data = data.sort_values(by='Date')
data = data.reset_index(drop=True)
data = data.dropna()

# Calculate the Black-Scholes-Merton prices of European put and call options, and check for put-call parity


def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    eu_call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return eu_call_price


def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    eu_put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return eu_put_price


data["eu_call"] = black_scholes_call(data["Close"], 305, 20/365, data["rf"], data["Volatility"])
data["eu_put"] = black_scholes_put(data["Close"], 305, 20/365, data["rf"], data["Volatility"])


def check_put_call_parity(call_price, put_price, stock_price, strike_price, time_to_maturity, risk_free_rate):
    tolerance = 0.000001
    lhs = call_price - put_price
    rhs = stock_price - strike_price * np.exp(-risk_free_rate * time_to_maturity)
    diff = abs(lhs - rhs)

    return diff < tolerance


data["check"] = check_put_call_parity(data["eu_call"], data["eu_put"], data["Close"], 305, 20/365, data["rf"])
print(data['check'].value_counts())

plt.plot(data["eu_call"])
plt.plot(data["eu_put"])
plt.grid()
plt.title(f'{ticker}' " put and call prices")
plt.show()

# Compute the Delta of an option


def black_scholes_delta(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if option_type == "call":
        return norm.cdf(d1)
    elif option_type == "put":
        return norm.cdf(d1) - 1


data["delta_call"] = black_scholes_delta(data["Close"], 305, 20/365, data["rf"], data["Volatility"], option_type="call")
data["delta_put"] = black_scholes_delta(data["Close"], 305, 20/365, data["rf"], data["Volatility"], option_type="put")

plt.plot(data["delta_call"])
plt.plot(data["delta_put"])
plt.grid()
plt.title(f'{ticker}' " delta for put and call options")
plt.show()
