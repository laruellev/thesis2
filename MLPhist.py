import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

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

for i in range(len(data)):
    data["Datetime"].iloc[i] = dt.datetime(data["Year"][i], data["Month"][i], data["Day"][i])
plt.plot(data["Datetime"], data["shares_purchased"], label="Shares bought or sold")
plt.plot(data["Datetime"], data["shares_tot"], label="Total shares")
plt.grid()
plt.ylabel("Shares")
plt.xlabel("Time")
plt.legend()
plt.show()

plt.plot(data["Datetime"], data["interest"], label="Cost of interest")
plt.plot(data["Datetime"], data["cost"], label="Cost of shares")
plt.plot(data["Datetime"], data["cost_tot"], label="Total cost")
plt.grid()
plt.ylabel("Cost in thousand $")
plt.xlabel("Time")
plt.legend()
plt.show()

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(data["Datetime"], data["shares_purchased"])
axs[0, 0].set_title('Shares purchased')
axs[0, 0].set_xticks([])
axs[0, 1].plot(data["Datetime"], data["shares_tot"], 'tab:orange')
axs[0, 1].set_title('Shares in total')
axs[0, 1].set_xticks([])
axs[1, 0].plot(data["Datetime"], data["cost"], 'tab:green')
axs[1, 0].set_title('Cost of shares in thousand USD')
axs[1, 1].plot(data["Datetime"], data["interest"], 'tab:red')
axs[1, 1].set_title('Cost of interest in thousand USD')

for ax in axs.flat:
    ax.set(xlabel='Time')
    ax.tick_params(labelrotation=45)

data_mlp = data[["Close", "LogReturn", "Volatility", "rf", "shares_purchased"]]
# removed: "eu_call", "Expiry", "delta_call"
data_mlp["Close"] = data_mlp["Close"]/100
data_mlp["LogReturn"] = data_mlp["LogReturn"]*10
data_mlp["rf"] = data_mlp["rf"]*10
# data_mlp["eu_call"] = data_mlp["eu_call"]/100
# data_mlp["Expiry"] = data_mlp["Expiry"]/1000
data_mlp["shares_purchased"] = data_mlp["shares_purchased"]/10000

# Fill diagonal and upper half with NaNs
corr = data_mlp.corr()
corr.style.background_gradient(cmap='coolwarm')
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
corr[mask] = np.nan
(corr
 .style
 .background_gradient(cmap='coolwarm', axis=None, vmin=-1, vmax=1)
 .highlight_null(color='#f1f1f1')  # Color NaNs grey
 .format(precision=2))
plt.matshow(corr)
cb = plt.colorbar()
cb.ax.tick_params()
plt.show()

# 0. get ready
print("Begin scikit neural network regression example ")
print("Predict shares purchased from other financial data")
np.random.seed(82)

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

