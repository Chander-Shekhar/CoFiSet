import numpy as np
import pandas as pd

data = pd.read_table('../dataset/udata.txt', delim_whitespace=True, header = None, parse_dates=False)
dataset = data.values[:, :2]
feedback = data.values[:, 2]
observed_data = dataset[feedback > 3]
print(observed_data)
users = np.unique(observed_data[:, 0])
train = []
test = []
tmp = []
for user in range(942):
	indices = (np.where(observed_data[:, 0] == users[user]))[0]
	tmp = observed_data[indices][:, 1]
	train_data = np.random.choice(tmp, size=tmp.shape[0]//2, replace=False)
	train.append(train_data)
	test.append(np.array([item for item in tmp if item not in list(train_data)]))

