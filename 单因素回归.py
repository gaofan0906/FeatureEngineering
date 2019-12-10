from scipy import stats
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import Imputer
from scipy.stats import chi2_contingency
from statsmodels.stats.anova import anova_lm

data_path = ('./0903data.csv')
df = pd.read_csv(data_path)

df["性别"].replace('男', '1', inplace=True)
df["性别"].replace('女', '0', inplace=True)


# 单因素回归
df["性别"] = df["性别"].astype(int)
model = sm.Logit(endog=df["性别"], exog=df[["血红蛋白浓度"]]).fit()
print(model.summary())
# 多因素回归
model = sm.OLS(endog=df["性别"], exog=df[["血红蛋白浓度"]]).fit()
print(model.summary())
# P<0.05则认为自变量具有统计学意义


# 差异性检验
# 两个离散
# 统计
count00 = 0
count01 = 0
count02 = 0
count03 = 0
count10 = 0
count11 = 0
count12 = 0
count13 = 0
l = zip(list(df['性别']), list(df["心功能分级(NYHA)"]))
for i in l:
    if i == (0, 0):
        count00 += 1
    elif i == (0, 1):
        count01 += 1
    elif i == (0, 2):
        count02 += 1
    elif i == (0, 3):
        count03 += 1
    elif i == (1, 0):
        count10 += 1
    elif i == (1, 1):
        count11 += 1
    elif i == (1, 2):
        count12 += 1
    elif i == (1, 3):
        count13 += 1
# 统计值中包含0就不能用卡方检验
a = [count10,
     count11,
     count12,
     count13]
c = [count00,
     count01,
     count02,
     count03]
kf_datad = np.array([a, c])
# 卡方检验
chi2_contingency(kf_datad)

a = [19, 14, 12, 3]
b = [7, 4, 3, 2]
c = [5, 3, 2, 1]
d = [3, 5, 2, 1]
kf_datad = np.array([a, b, c, d])
# 卡方检验
chi2_contingency(kf_datad)

# 因变量是离散的，自变量是连续的
# 非参数检验
# 性别，血红蛋白浓度
a = [118,
     187.7,
     243.9,
     123,
     191.5,
     112,
     106.6,
     169.8,
     140,
     135.7,
     191,
     99,
     177.8,
     194,
     218.8,
     129.5,
     150,
     150,
     209,
     142,
     96,
     130.5,
     99.3,
     93,
     148.3,
     107,
     153,
     127,
     99,
     108,
     138.7]
b = [196,
     120.4,
     168.8,
     154,
     136.9,
     130.1,
     116,
     195.1,
     188,
     195.9,
     107,
     132,
     132,
     149.3,
     162,
     144.8,
     134,
     155,
     167.8,
     203.2,
     124.5,
     117,
     151,
     198,
     91.3,
     246.5,
     262,
     107.7,
     141.9,
     100.2,
     154,
     145,
     157,
     118,
     144,
     171.4,
     147.5,
     130.2,
     128.5,
     151,
     157,
     105,
     141,
     116,
     125,
     183.9,
     211.1,
     157,
     251,
     137.2,
     160.4,
     105.1,
     137.4,
     182,
     193]
# 秩和检验
stats.ranksums(a, b)

# 性别 红细胞计数（符合正态分布）
a = [3.354,
     7.235,
     8.763,
     5.5,
     8.354,
     6.19,
     5.418,
     6.662,
     5.18,
     4.22,
     6.3,
     5.8,
     6.157,
     8.02,
     6.662,
     5.567,
     5.95,
     6.23,
     6.26,
     5.65,
     5.87,
     4.574,
     4.714,
     4.21,
     5.811,
     3.93,
     5.49,
     4.26,
     4.58,
     4.31,
     5.192]
b = [7.15,
     5.507,
     5.825,
     5.5,
     5.144,
     4.286,
     4.26,
     7.611,
     5.84,
     5.824,
     4.5,
     3.44,
     4.739,
     5.274,
     5.39,
     4.449,
     4.34,
     5.69,
     5.449,
     6.98,
     4.872,
     5.49,
     5.15,
     6.81,
     2.546,
     7.613,
     7.95,
     4.309,
     5.71,
     3.49,
     5.93,
     4.88,
     6.54,
     4.31,
     5.4,
     5.727,
     5.186,
     5.739,
     4.748,
     6.72,
     5.99,
     3.86,
     5.47,
     4.27,
     4.62,
     5.632,
     6.529,
     6.54,
     7.57,
     5.639,
     4.975,
     6.338,
     5.345,
     5.91,
     7]
# 服从正态分布，t检验
# 独立两样本t检验
stats.ttest_ind(a, b, equal_var=True, nan_policy='omit')


# 多分类
# 参数方法
stats.f_oneway(a, b, c, ...)

# 非参数方法
stats.kruskal(a, b, c,..., nan_policy='omit')

# 相关性
# 二元值和连续值之间的相关性
stats.pointbiserialr(df['性别'], df['血红蛋白浓度'])

stats.pearsonr(df['血红蛋白浓度'],df['红细胞计数'])

stats.spearmanr(df['血红蛋白浓度'],df['红细胞计数'])


