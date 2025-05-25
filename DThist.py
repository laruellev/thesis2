import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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

data["label"] = np.nan
for i in range(len(data)):
    if data["shares_purchased"][i] > 0:
        data["label"][i] = 1  # buy
    elif data["shares_purchased"][i] < 0:
        data["label"][i] = 0  # sell

x = data[["Close", "LogReturn", "Volatility", "rf", "eu_call", "Expiry", "delta_call"]]
y = data[["label"]]

np.random.seed(82)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

# Create Decision Tree classifier object
clf = DecisionTreeClassifier()

# Train Decision Tree classifier
clf = clf.fit(x_train,y_train)

# Predict the response for test dataset
preds = clf.predict(x_test)

# Model accuracy
y_test_acc = y_test.reset_index(drop=True)
errs = []
for i in range(len(y_test_acc)):
    if float(y_test_acc["label"][i]) == float(preds[i]):
        errs.append(True)
    if float(y_test_acc["label"][i]) != float(preds[i]):
        errs.append(False)
acc = errs.count(True)/len(errs) * 100
print("model accuracy:", acc, "%")


'''
The decision tree model is more robust to noise since it does not matter the value of the shares bought, only if it's 
bought or sold, therefore as long as the noise keeps a 'buy' decision as 'buy' (and a 'sell' decision as 'sell), 
there is no error introduced by the noise. 
'''
