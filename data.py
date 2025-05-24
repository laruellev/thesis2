import datetime as dt

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import yfinance as yf

from scipy.stats import norm

cwd = '/Users/vlr/PycharmProjects/thesis'
dl = '/Users/vlr/Downloads/Masters/python'

ticker = "NVDA"
dat = yf.Ticker(ticker)  # stock
expiry_dates = dat.options
T = 2
timeframe = f'{T}'+"y"

# Get stock historical data
hist = pd.DataFrame(dat.history(period=timeframe))  # Stock opening and closing prices for past 5 years
hist["GrossReturn"] = hist.Close.pct_change()  # Percentage change from one day to the other
hist["LogReturn"] = np.log(hist.Close) - np.log(hist.Close.shift(1))  # Logarithmic return
hist = hist.iloc[1:]
hist["Volatility"] = hist.LogReturn.rolling(window=20).std()*np.sqrt(252)
# 20 daily returns = 1 trading month, annualized
hist = hist.dropna()
'''
# Plot the stock price over time
plt.plot(hist["Close"])
plt.grid()
plt.ylabel(f'{ticker}' " stock price")
plt.show()

# Plot the log return over time
plt.plot(hist["LogReturn"])
plt.grid()
plt.ylabel(f'{ticker}' " log return")
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
plt.ylabel("Risk free rate in %")
plt.grid()
plt.show()
'''
# Combine all the date into one dataframe
hist["Date"] = hist.index.strftime('%m/%d/%Y')
hist = hist.reset_index(drop=True)
rf = pd.read_csv(dl+"/10y_US_bond.csv")
rf["Price"] = rf["Price"]/100
rf = rf.rename(columns={"Date": "Date", "Price": "rf"})
rf = rf.drop(['Open', 'High', 'Low', 'Change %'], axis=1)
data = pd.merge(hist, rf, how='left', on=['Date'])
data["Date"] = data["Date"].str.split("/")
for i in range(len(data)):
    data["Date"].iloc[i] = data["Date"][i][2] + data["Date"][i][0] + data["Date"][i][1]
data = data.sort_values(by='Date')
data = data.dropna()
data = data.reset_index(drop=True)

data_to_latex = data.truncate(after=10)
data_to_latex = data_to_latex.iloc[:, -6:]
print(data_to_latex.to_latex(index=False))

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


data["eu_call"] = np.nan
for i in range(len(data)):
    data["eu_call"].iloc[i] = black_scholes_call(data["Close"][i], np.average(data["Close"]),
                                                 T, data["rf"][i], data["Volatility"][i])
data["eu_put"] = np.nan
for i in range(len(data)):
    data["eu_put"].iloc[i] = black_scholes_put(data["Close"][i], np.average(data["Close"]),
                                               T, data["rf"][i], data["Volatility"][i])


def check_put_call_parity(call_price, put_price, stock_price, strike_price, time_to_maturity, risk_free_rate):
    tolerance = 0.000001
    lhs = call_price - put_price
    rhs = stock_price - strike_price * np.exp(-risk_free_rate * time_to_maturity)
    diff = abs(lhs - rhs)

    return diff < tolerance


data["check"] = np.nan
for i in range(len(data)):
    data["check"].iloc[i] = check_put_call_parity(data["eu_call"][i], data["eu_put"][i], data["Close"][i],
                                      np.average(data["Close"]), T, data["rf"][i])
print(data['check'].value_counts())
'''
plt.plot(data["eu_call"], label=f'{ticker}' " European call")
plt.plot(data["eu_put"], label=f'{ticker}' " European put")
plt.legend()
plt.grid()
plt.title(f'{ticker}' " put and call prices")
plt.show()
'''

# Compute the amount of days to expiration
data["Year"] = data["Date"].str[0:4].astype(int)
data["Month"] = data["Date"].str[4:6].astype(int)
data["Day"] = data["Date"].str[6:8].astype(int)
data["Datetime"] = np.nan
for i in range(len(data)):
    data["Datetime"].iloc[i] = dt.datetime(data["Year"][i], data["Month"][i], data["Day"][i])
expiry = data["Datetime"][len(data)-1]
data["Expiry"] = np.nan
for i in range(len(data)):
    data["Expiry"].iloc[i] = np.busday_count(data["Datetime"][i].date(), expiry.date())

# Compute the Delta of an option


def black_scholes_delta(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if option_type == "call":
        return norm.cdf(d1)
    elif option_type == "put":
        return norm.cdf(d1) - 1


data["delta_call"] = np.nan
for i in range(len(data)):
    data["delta_call"].iloc[i] = black_scholes_delta(data["Close"][i], np.average(data["Close"]), data["Expiry"][i]/365,
                                                     data["rf"][i], data["Volatility"][i], option_type="call")
data["delta_put"] = np.nan
for i in range(len(data)):
    data["delta_put"].iloc[i] = black_scholes_delta(data["Close"][i], np.average(data["Close"]), data["Expiry"][i]/365,
                                                    data["rf"][i], data["Volatility"][i], option_type="put")

plt.plot(data["delta_call"], label=f'{ticker}' " EU call Delta")
plt.plot(data["delta_put"], label=f'{ticker}' " EU put Delta")
plt.legend()
plt.grid()
plt.title(f'{ticker}' " put and call options' Delta")
plt.show()

a = 0

# data.to_csv(dl+"/thesis_data.csv", index=False)
