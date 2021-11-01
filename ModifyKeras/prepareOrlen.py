
# first neural network with keras tutorial
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import datetime


def getdata1D(name='pkn_d.csv', timesteps=7, ytargetname='Zamkniecie'):

    data = pd.read_csv(name)
    columnsName = list(data.columns)
    Z = columnsName.index(ytargetname)
    scaler = MinMaxScaler()


    # print(data.head())

    data['Data'] = data['Data'].apply(lambda x: pd.Timestamp(x).weekday())
    # print(data.head())
    data = data.to_numpy()
    # print(data)

    scaler.fit(data)
    # print(scaler.data_max_)
    # print(scaler.data_min_)
    data = scaler.transform(data)

    xdata = []
    ydata = []
    N, input_dim = data.shape

    for i, seq in enumerate(data):
        # print(i, seq)
        if i <= N - timesteps - 1: # -1 because w need one row for y
            b = []
            for j in range(input_dim):

                a = []
                for k in range(timesteps):
                    a.append(data[k+i][j])
                    # print(k+i, end=' ')

                b.append(a)

            ydata.append(data[timesteps+i][Z])
            # print(" => ",timesteps+i)
            xdata.append(b)

    return np.array(xdata), np.array(ydata)

if __name__ == "__main__":
    timesteps = 7
    X, Y = getdata1D('pkn_d.csv', timesteps, 'Zamkniecie')

