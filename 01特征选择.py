# 方差选择
# 计算所有特征的方差值并排序
# 连续变量的方差值
import pandas as pd
import numpy as np

data_path = './data0930.csv'
df = pd.read_csv(data_path)
df.drop(columns='Unnamed: 0', inplace=True)


# 连续变量计算方差并排序
# 得到连续变量
def ContinuousVariable(dataframe):
    res = pd.DataFrame()
    n = 0
    for i in range(dataframe.shape[1]):
        if type(dataframe.iloc[:, i][0]) is np.float64:
            res.insert(n, dataframe.iloc[:, i].name, dataframe.iloc[:, i])
            n += 1
        else:
            pass
    return res


df3 = ContinuousVariable(df)


# 计算方差排序
def varsort(dataframe):
    dict = {}
    for i in range(dataframe.shape[1]):
        dict.update({dataframe.iloc[:, i].name: dataframe.iloc[:, i].var()})
    return sorted(dict)


di = varsort(df3)

from sklearn.feature_selection import VarianceThreshold

# 方差选择法，返回值为特征选择后的数据,无排序
# 参数threshold为方差的阈值
l=VarianceThreshold(threshold=1).fit_transform(df3)


# 得到离散变量并卡方检验
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
# 卡方检验：离散特征
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# 选择K个最好的特征，返回选择特征后的数据
SelectKBest(chi2, k=10).fit_transform(df2, df2.iloc[:,1])

# 相关系数：全特征
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from numpy import vstack, array, nan,fromiter,asarray
import numpy as np

# 选择K个最好的特征，返回选择特征后的数据
# 第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，
# 输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
# 参数k为选择的特征个数

SelectKBest(lambda X, Y: tuple(array(list(map(lambda x: pearsonr(x, Y), X.T)),dtype='float64').T), k=10).fit_transform(df, df2.iloc[:,0])

# 互信息：离散特征
from sklearn.feature_selection import SelectKBest
from minepy import MINE
# 由于MINE的设计不是函数式的，定义mic方法将其为函数式的，
# 返回一个二元组，二元组的第2项设置成固定的P值0.5
def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return (m.mic(), 0.5)


# 选择K个最好的特征，返回特征选择后的数据
SelectKBest(lambda X, Y: tuple(array(list(map(lambda x: mic(x, Y), X.T)),dtype='float64').T), k=10).fit_transform(df2, df2.iloc[:,0])

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# 递归特征消除法，返回特征选择后的数据
# 参数estimator为基模型
# 参数n_features_to_select为选择的特征个数
RFE(estimator=LogisticRegression(), n_features_to_select=10).fit_transform(df, df2.iloc[:,0])

import statsmodels.api as sm
# 单因素回归
# 多因素回归
# 多元线性回归

model = sm.OLS(endog=df2.iloc[:,0], exog=df).fit()
model.pvalues
print(model.summary())
# P<0.05则认为自变量具有统计学意义

# 逻辑回归
# 只能是二分类的回归
model = sm.Logit(endog=df2.iloc[:,0], exog=df).fit()
print(model.summary())


