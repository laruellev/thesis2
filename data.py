import pandas as pd
import datetime
import yfinance as yf

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

puts.to_csv(dl+"/NVDA_puts.csv")
calls.to_csv(dl+"/NVDA_calls.csv")
hist.to_csv(dl+"/NVDA_stock.csv")
