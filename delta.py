import pandas as pd
from data import dl, cwd
import numpy as np

n_shares = 100000
data = pd.read_csv(dl+"/thesis_data.csv")

data["shares"] = data["delta_call"]*n_shares
data["shares_tot"] = np.nan
for i in range(len(data)):
    data_truncated0 = data.truncate(after=i)
    data["shares_tot"][i] = data["shares"].sum()
data["cost"] = data["shares"]*data["Close"]/1000
data["interest"] = np.nan
data["cost_tot"] = np.nan
for i in range(len(data)):
    if i == 0:
        data["cost_tot"][i] = data["cost"][i]
        data["interest"][i] = data["cost_tot"][i]*(np.exp((data["rf"][i]/365))-1)
    if i > 0:
        data_truncated1 = data.truncate(after=i)
        data_truncated2 = data.truncate(after=i-1)
        data["cost_tot"].iloc[i] = data_truncated1["cost"].sum() + data_truncated2["interest"].sum()
        data["interest"].iloc[i] = data["cost_tot"][i]*(np.exp((data["rf"][i]/365))-1)

a = 0

