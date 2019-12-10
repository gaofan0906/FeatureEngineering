from scipy import stats
import numpy as np

np.random.seed(12345678)
# 夏皮罗维尔克检验法 (Shapiro-Wilk) 用于检验参数提供的一组小样本数据线是否符合正态分布，
# 检验小样本数据（数据量n<50）是否服从正态分布。
x = stats.norm.rvs(loc=5, scale=10, size=80)  # loc为均值，scale为方差
# shapiro检验
print(stats.shapiro(y))
'''
    Returns
    -------
    W : float
        The test statistic.
    p-value : float
        The p-value for the hypothesis test.
        '''
# 运行结果：(0.9654011726379395, 0.029035290703177452)
# 小于0.05，拒绝原假设，x不服从正态分布

# 科尔莫戈罗夫检验(Kolmogorov-Smirnov test)，检验样本数据是否服从某一分布。
# 仅适用于连续分布的检验。下例中用它检验正态分布。
y = stats.norm.rvs(loc=0, scale=1, size=300)
print(stats.kstest(y, 'norm'))
# p>0.05,接受原假设即y服从正态分布（接受只是没有足够的证据拒绝）

# 方差齐性检验
# 方差反映了一组数据与其平均值的偏离程度
# 方差齐性检验用以检验两组或多组数据与其均值偏离程度是否存在差异
# 也是很多检验和算法的先决条件。
rvs1 = stats.norm.rvs(loc=5, scale=10, size=500)
rvs2 = stats.norm.rvs(loc=25, scale=9, size=500)
print(stats.levene(rvs1, rvs2))

# 图形描述相关性
# 线性正相关一般形成由左下到右上的图形；
# 负相关则是从左上到右下的图形，还有一些非线性相关也能从图中观察到。
import statsmodels.api as sm
import matplotlib.pyplot as plt

data = sm.datasets.ccard.load_pandas().data
plt.scatter(data['INCOMESQ'], data['INCOME'])  # 散点图
plt.show()

# 皮尔森相关系数
# 俩变量之间线性相关程度的统计量，用它来分析正态分布的两个连续型变量之间的相关性。
# 常用于分析自变量之间，以及自变量和因变量之间的相关性。
a = np.random.normal(0, 1, 100)
b = np.random.normal(2, 2, 100)
print(stats.pearsonr(a, b))
# 第一个值为相关系数表示线性相关程度，其取值范围在[-1,1]，绝对值越接近1，说明两个变量的相关性越强，绝对值越接近0说明两个变量的相关性越差。
# p-value，统计学上，一般当p-value<0.05时，可以认为两变量存在相关性。


# 非正态资料的相关分析
# 斯皮尔曼等级相关系数，用于评价顺序变量间的线性相关关系，在计算过程中，只考虑变量值的顺序（rank, 秩或称等级），而不考虑变量值的大小。
print(stats.spearmanr([1, 2, 3, 4, 5], [5, 6, 7, 8, 7]))
# 第一个值为相关系数表示线性相关程度，本例中correlation趋近于1表示正相关。
# 第二个值为p-value，p-value越小，表示相关程度越显著。

# 单样本T检验
# 用于检验数据是否来自一致均值的总体
rvs = stats.norm.rvs(loc=5, scale=10, size=(100, 2))
# [1, 5]分别对两列估计的均值
print(stats.ttest_1samp(rvs, [1, 5]))
# 第一列小于0.05拒绝假设。第二列大于0.05，不拒绝假设，服从正态分布

#  两独立样本T检验
# 比较两组数据是否来自于同一正态分布的总体
rvs1 = stats.norm.rvs(loc=5, scale=10, size=500)
rvs2 = stats.norm.rvs(loc=6, scale=10, size=500)
print(stats.ttest_ind(rvs1, rvs2))
# 第一个结果是统计量
# p比0.05大，不拒绝假设，说明两组数据之间无差异

# 配对样本T检验
rvs1 = stats.norm.rvs(loc=5, scale=10, size=500)
rvs2 = (stats.norm.rvs(loc=5, scale=10, size=500) + stats.norm.rvs(scale=0.2, size=500))
print(stats.ttest_rel(rvs1, rvs2))

# 单因素方差分析：是检验由单一因素影响的多组样本某因变量的均值是否有显著差异。
# 方差分析（F检验）主要是考虑各组之间的均数差别。
a = [47, 56, 46, 56, 48, 48, 57, 56, 45, 57]  # 分组1
b = [87, 85, 99, 85, 79, 81, 82, 78, 85, 91]  # 分组2
c = [29, 31, 36, 27, 29, 30, 29, 36, 36, 33]  # 分组3
print(stats.f_oneway(a, b, c))

#  多因素方差分析
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import pandas as pd

X1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
      2, 2]
X2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
      2, 2]
Y = [76, 78, 76, 76, 76, 74, 74, 76, 76, 55, 65, 90, 65, 90, 65, 90, 90, 79, 70, 90, 88, 76, 76, 76, 56, 76, 76, 98, 88,
     78, 65, 67, 67, 87, 78, 56, 54, 56, 54, 56]

data = {'T': X1, 'G': X2, 'L': Y}
df = pd.DataFrame(data)
formula = 'L~T+G+T:G'  # 公式
model = ols(formula, df).fit()
print(anova_lm(model))
# 上述程序定义了公式，公式中，"~"用于隔离因变量和自变量，”+“用于分隔各个自变量， ":"表示两个自变量交互影响。
# 从返回结果的P值可以看出，X1和X2的值组间差异不大，而组合后的T:G的组间有明显差异。


# 卡方检验：非参数检验方法
# 主要是比较理论频数和实际频数的吻合程度。常用于特征选择
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

np.random.seed(12345678)
data = np.random.randint(2, size=(40, 3))  # 2个分类，50个实例，3个特征
data = pd.DataFrame(data, columns=['A', 'B', 'C'])
contingency = pd.crosstab(data['A'], data['B'])  # 建立列联表
print(chi2_contingency(contingency))  # 卡方检验

# 多元线性回归
import statsmodels.api as sm

data = sm.datasets.ccard.load_pandas().data
model = sm.OLS(endog=data['AVGEXP'], exog=data[['AGE', 'INCOME', 'INCOMESQ', 'OWNRENT']]).fit()
print(model.summary())
# P<0.05则认为自变量具有统计学意义

# 逻辑回归
# 当因变量Y为2分类变量（或多分类变量时）可以用相应的logistic回归分析各个自变量对因变量的影响程度。
import statsmodels.api as sm

data = sm.datasets.ccard.load_pandas().data
data['OWNRENT'] = data['OWNRENT'].astype(int)
model = sm.Logit(endog=data['OWNRENT'], exog=data[['AVGEXP', 'AGE', 'INCOME', 'INCOMESQ']]).fit()
print(model.summary())
