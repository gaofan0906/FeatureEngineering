import pandas as pd
import numpy as np

data_path = './data0930.csv'
df = pd.read_csv(data_path)
df.drop(columns='Unnamed: 0', inplace=True)


# 得到离散变量
def DiscreteVariable(dataframe):
    res = pd.DataFrame()
    n = 0
    for i in range(dataframe.shape[1]):

        if type(dataframe.iloc[:, i][0]) is np.int64:
            res.insert(n, df.iloc[:, i].name, df.iloc[:, i])
            n += 1
        else:
            pass
    return res


df2 = DiscreteVariable(df)


# 得到连续变量
def ContinuousVariable(dataframe):
    res = pd.DataFrame()
    n = 0
    for i in range(dataframe.shape[1]):

        if type(dataframe.iloc[:, i][0]) is np.float64:
            res.insert(n, df.iloc[:, i].name, df.iloc[:, i])
            n += 1
        else:
            pass
    return res


df3 = ContinuousVariable(df)
