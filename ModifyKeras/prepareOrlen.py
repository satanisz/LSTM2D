
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

    # for test
    if False:
        data = np.array(range(20))
        data = data.reshape(-1,1)
        Z = 0
        print(data)
    # for test

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



def getdata2D(name='pkn_d.csv', timesteps=7, ytargetname='Zamkniecie'):

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

    # for test
    if False:
        data = np.array([[x, x] for x in range(20)])
        data = data.reshape(-1,2)
        Z = 1
        print(data)
    # for test

    # for test
    if False:
        data = np.array([x for x in range(20)])
        data = data.reshape(-1,1)
        Z = 0
        print(data)
    # for test


    xdata = [] # now x data is 2D  timestemp 3->5; 4->7  n -> n+n-1
    # 6 7 8 9 10 11
    # 5 6 7 8 9 10
    # 4 5 6 7 8 9
    # 3 4 5 6 7 8
    # 2 3 4 5 6 7
    # 1 2 3 4 5 6
    timesteps = int((timesteps+1)/2) # reverse
    ydata = []
    N, input_dim = data.shape

    for i, seq in enumerate(data):
        # print(i, seq)
        if i <= N - 2*timesteps: # -1 because w need one row for y
            b = []
            for j in range(input_dim):

                a1 = []
                for m in range(timesteps):
                    a0 = []
                    for n in range(timesteps):
                        a0.append(data[n+m+i][j])
                    a1.append(a0)
                    # print(k+i, end=' ')

                b.append(a1)

            ydata.append(data[timesteps+i][Z])
            # print(" => ",timesteps+i)
            xdata.append(b)

    return np.array(xdata), np.array(ydata)

if __name__ == "__main__":
    timesteps = 7
    # X, Y = getdata1D('pkn_d.csv', timesteps, 'Zamkniecie')

    X, Y = getdata2D('pkn_d.csv', timesteps, 'Zamkniecie')

