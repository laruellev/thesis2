import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np
from data import ticker

dat = yf.Ticker(ticker)  # stock
expiry_dates = dat.options

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
