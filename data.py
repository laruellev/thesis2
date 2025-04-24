import pandas as pd
import datetime
import numpy as np
import yfinance as yf
from scipy.stats import norm
from itertools import product

cwd = '/Users/vlr/PycharmProjects/thesis'
dl = '/Users/vlr/Downloads/Masters/python'

dat = yf.Ticker("NVDA")  # Nvidia stock
expiry_dates = dat.options  # Nvidia stock options expiry dates
hist = pd.DataFrame(dat.history(period="10y"))  # Nvidia stock opening and closing prices for past 10 years

# Extract historical data for Nvidia stock options: call for each expiry date

calls = {}
for i in range(len(expiry_dates)):
    key = str('call_'+str(expiry_dates[i]))
    calls[key] = pd.DataFrame(dat.option_chain(date=expiry_dates[i]).calls)
    calls[key]["expiryDate"] = str(expiry_dates[i])
    calls[key]["type"] = "call_option"

    end = datetime.date(
        int(str.split(expiry_dates[i], "-")[0]),
        int(str.split(expiry_dates[i], "-")[1]),
        int(str.split(expiry_dates[i], "-")[2])
    )
    start = datetime.date(2025, 4, 25)
    calls[key]["T"] = str(np.busday_count(start, end))
calls = pd.concat(calls.values(), ignore_index=True)


# Extract historical data for Nvidia stock options: put for each expiry date

puts = {}
for i in range(len(expiry_dates)):
    key = str('call_'+str(expiry_dates[i]))
    puts[key] = pd.DataFrame(dat.option_chain(date=expiry_dates[i]).puts)
    puts[key]["expiryDate"] = str(expiry_dates[i])
    puts[key]["type"] = "put_option"

    end = datetime.date(
        int(str.split(expiry_dates[i], "-")[0]),
        int(str.split(expiry_dates[i], "-")[1]),
        int(str.split(expiry_dates[i], "-")[2])
    )
    start = datetime.date(2025, 4, 25)
    puts[key]["T"] = str(np.busday_count(start, end))
puts = pd.concat(puts.values(), ignore_index=True)

'''
puts.to_csv(dl+"/NVDA_puts_20250425.csv")
calls.to_csv(dl+"/NVDA_calls_20250425.csv")
hist.to_csv(dl+"/NVDA_stock_20250425.csv")
'''

# ATTEMPT AT GENERATING DECENT SYNTHETIC DATA = FAIL #


def rem_dupl(lst):
    return list(dict.fromkeys(lst))


def black_scholes_price(S, K, r, T, sigma):  # arbitrage-free price of a call option
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price


K = np.array(rem_dupl(sorted(calls["strike"].values.tolist())))
sigma = np.array(rem_dupl(sorted(calls["impliedVolatility"].tolist()))[1:])[0::5]
T = np.array(list(map(int, rem_dupl(calls["T"].values.tolist())[1:])))
r = np.array([4.32/100])  # https://tradingeconomics.com/united-states/government-bond-yield
S = np.array([106.43])  # current underlying stock price on Apr 24, 2025

option_prices = pd.DataFrame(
  product(S, K, r, T, sigma),
  columns=["S", "K", "r", "T", "sigma"]
)

option_prices["black_scholes"] = black_scholes_price(
  option_prices["S"].values,
  option_prices["K"].values,
  option_prices["r"].values,
  option_prices["T"].values,
  option_prices["sigma"].values
)

option_prices = (option_prices
  .assign(
    observed_price=lambda x: (
      x["black_scholes"] + np.random.normal(scale=0.5)
    )
  )
)

a = 0
