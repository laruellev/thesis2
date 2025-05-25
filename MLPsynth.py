import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from MLPhist import data_mlp as data_mlp_hist

cwd = '/Users/vlr/PycharmProjects/thesis'
dl = '/Users/vlr/Downloads/Masters/python'

n_shares = 100000
data = pd.read_csv(dl+"/thesis_data.csv")

data["shares"] = data["delta_call"]*n_shares
data["shares_purchased"] = np.nan
for i in range(len(data)):
    if i == 0:
        data["shares_purchased"][0] = data["delta_call"][0]*n_shares
    if i > 0:
        data["shares_purchased"][i] = (data["delta_call"][i]-data["delta_call"][i-1])*n_shares
data["shares_tot"] = np.nan
for i in range(len(data)):
    if i == 0:
        data["shares_tot"][i] = data["shares_purchased"][i]
    if i > 0:
        data["shares_tot"][i] = data.truncate(after=i)["shares_purchased"].sum()
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

data_mlp = data[["Close", "LogReturn", "Volatility", "rf", "shares_purchased"]]  # "eu_call", "Expiry", "delta_call"
data_mlp["Close"] = data_mlp["Close"]/100
data_mlp["LogReturn"] = data_mlp["LogReturn"]*10
data_mlp["rf"] = data_mlp["rf"]*10
# data_mlp["eu_call"] = data_mlp["eu_call"]/100
# data_mlp["Expiry"] = data_mlp["Expiry"]/1000
data_mlp["shares_purchased"] = data_mlp["shares_purchased"]/10000
noise = np.random.normal(0, 0.1, len(data_mlp))  # Gaussian noise
data_mlp["shares_purchased"] = data_mlp["shares_purchased"] + noise

for i in range(len(data)):
    data["Datetime"].iloc[i] = dt.datetime(data["Year"][i], data["Month"][i], data["Day"][i])
plt.plot(data_mlp_hist["shares_purchased"], label="Shares bought or sold")
plt.plot(data_mlp["shares_purchased"], label="Shares bought or sold with noise")
plt.grid()
plt.ylabel("Shares")
plt.xlabel("Time")
plt.legend()
plt.show()

corr = data_mlp.corr()


# 0. get ready
print("Begin scikit neural network regression example ")
print("Predict shares purchased from other financial data")
np.random.seed(69)
np.set_printoptions(precision=3, suppress=True)

# 1. load data
x = data_mlp[["Close", "LogReturn", "Volatility", "rf"]]  # "eu_call", "Expiry", "delta_call"
y = data_mlp[["shares_purchased"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size=.75)

# 2. create network
params = {'hidden_layer_sizes': [10, 10],
          'activation': 'relu', 'solver': 'adam',
          'alpha': 0.0, 'batch_size': 10,
          'random_state': 0, 'tol': 0.0001,
          'nesterovs_momentum': False,
          'learning_rate': 'constant',
          'learning_rate_init': 0.01,
          'max_iter': 1000, 'shuffle': True,
          'n_iter_no_change': 50, 'verbose': False}

net = MLPRegressor(**params)

# 3. train
train = net.fit(x_train, y_train)

# 4. predict
preds = net.predict(x_test)

# 5. model accuracy
y_test_acc = y_test.reset_index(drop=True)
errs = []
for i in range(len(y_test_acc)):
    if float(y_test_acc["shares_purchased"][i]) * float(preds[i]) > 0:
        errs.append(True)
    if float(y_test_acc["shares_purchased"][i]) * float(preds[i]) < 0:
        errs.append(False)
acc = errs.count(True)/len(errs) * 100
print("model accuracy:", acc, "%")
