import pandas as pd
import numpy as np

data_path = './data0930.csv'
df = pd.read_csv(data_path)
df.drop(columns='Unnamed: 0', inplace=True)


from sklearn.datasets import load_iris

# 导入IRIS数据集
iris = load_iris()

# 特征矩阵
p = iris.data

# 目标向量
i = iris.target
from sklearn.feature_selection import VarianceThreshold

# 方差选择法，返回值为特征选择后的数据
# 参数threshold为方差的阈值
l=VarianceThreshold(threshold=3).fit_transform(df)

from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from numpy import vstack, array, nan,fromiter,asarray
import numpy as np

# 选择K个最好的特征，返回选择特征后的数据
# 第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，
# 输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
# 参数k为选择的特征个数

a=SelectKBest(lambda X, Y: tuple(array(list(map(lambda x: pearsonr(x, Y), X.T)),dtype='float64').T), k=10).fit_transform(df, df.iloc[:,0])

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# 选择K个最好的特征，返回选择特征后的数据
SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target)

from sklearn.feature_selection import SelectKBest
from minepy import MINE


# 由于MINE的设计不是函数式的，定义mic方法将其为函数式的，
# 返回一个二元组，二元组的第2项设置成固定的P值0.5
def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return (m.mic(), 0.5)


# 选择K个最好的特征，返回特征选择后的数据
lll=SelectKBest(lambda X, Y: tuple(array(list(map(lambda x: mic(x, Y), X.T)),dtype='float64').T), k=10).fit_transform(df, df.iloc[:,0])

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# 递归特征消除法，返回特征选择后的数据
# 参数estimator为基模型
# 参数n_features_to_select为选择的特征个数
q=RFE(estimator=LogisticRegression(), n_features_to_select=10).fit_transform(df, df.iloc[:,0])

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

# 带L1惩罚项的逻辑回归作为基模型的特征选择
s=SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(df, df.iloc[:,0])

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier

# GBDT作为基模型的特征选择
x=SelectFromModel(GradientBoostingClassifier()).fit_transform(df, df.iloc[:,0])

from sklearn.decomposition import PCA

# 主成分分析法，返回降维后的数据
# 参数n_components为主成分数目
PCA(n_components=2).fit_transform(df)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.lda import LDA

# 线性判别分析法，返回降维后的数据
# 参数n_components为降维后的维数
z=LinearDiscriminantAnalysis(n_components=10).fit_transform(df, df.iloc[:,0])


