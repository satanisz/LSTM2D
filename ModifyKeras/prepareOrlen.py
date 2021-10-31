
# first neural network with keras tutorial
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import datetime

data = pd.read_csv('pkn_d.csv')
scaler = MinMaxScaler()
timesteps = 7

print(data.head())

data['Data'] = data['Data'].apply(lambda x: pd.Timestamp(x).weekday())
print(data.head())
data = data.to_numpy()
print(data)

print(scaler.fit(data))
print(scaler.data_max_)
print(scaler.data_min_)
data = scaler.transform(data)
features = []
labels = []

tdata = []
N, input_dim = data.shape

for i, seq in enumerate(data):
    print(i, seq)
    if i <= N - timesteps:
        b = []
        for j in range(input_dim):

            a = []
            for k in range(timesteps):
                a.append(data[k+i][j])

            b.append(a)

        tdata.append(b)

