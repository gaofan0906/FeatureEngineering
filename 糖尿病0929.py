import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from scipy.stats import chi2_contingency
from scipy import stats
import statsmodels.api as sm

data_path = './data0929.csv'
df = pd.read_csv(data_path)

# 数据
# 删除分类样本极不平衡的特征
# 删除填充率过低的特征，特征比较少，在外面处理

# 数据类型的确定
# 缺失值处理
# 在外面进行删除
# 代码填充
for i in range(df.shape[1]):
    # 判断是否包含空值
    if not all(~df.iloc[:, i].isna()):
        list_name = df.columns.values.tolist()
        df[list_name[i]] = df[list_name[i]].astype('float')
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        X = df.iloc[:, i].values
        Y = X.reshape(-1, 1)
        imp.fit(Y)
        df.iloc[:, i] = imp.transform(Y)
    else:
        pass


# 差异性检验
# 针对分类变量

# 差异性检验
# 两个离散
# 统计
count00 = 0
count01 = 0
count02 = 0

count10 = 0
count11 = 0
count12 = 0

l = zip(list(df['性别']), list(df["病程_分段（年）"]))
for i in l:
    if i == (0, 0):
        count00 += 1
    elif i == (0, 1):
        count01 += 1
    elif i == (0, 2):
        count02 += 1

    elif i == (1, 0):
        count10 += 1
    elif i == (1, 1):
        count11 += 1
    elif i == (1, 2):
        count12 += 1

# 统计值中包含0就不能用卡方检验
a = [count10,
     count11,
     count12,
     ]
b = [count00,
     count01,
     count02,
     ]
kf_datad = np.array([a, b])
# 卡方检验
chi2_contingency(kf_datad)

# 连续
# 正态分布检验
t, p = stats.shapiro(df['信息采集时年龄'])
# p小于0.05拒绝，不服从正态分布

# 秩和检验

grouped = df['信息采集时年龄'].groupby(df['性别'])
c = []
d = []
for group_name, group_data in grouped:
    if group_name == 0:
        for i in group_data:
            c.append(i)
    else:
        for i in group_data:
            d.append(i)

stats.ranksums(c, d)
# p>0.05无组间差异

# 多元线性回归
list_name = df.columns.values.tolist()
model = sm.OLS(endog=df['性别'], exog=df[['发病年龄', '葡萄糖.FPG ', '葡萄糖.F2hPG ', 'MA (mgl/L)']]).fit()
print(model.summary())
# P<0.05则认为自变量具有统计学意义

# 逻辑回归
# 只能是二分类的回归
model = sm.Logit(endog=df['性别'], exog=df[['发病年龄', '葡萄糖.FPG ', '葡萄糖.F2hPG ', '病程（年）']]).fit()
print(model.summary())

