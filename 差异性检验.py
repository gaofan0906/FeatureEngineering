from scipy import stats
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer

data_path = ('./0903datanew.csv')
df = pd.read_csv(data_path)
# (1389, 139)

# 删除列中只有一个分类的指标

for i in range(df.shape[1]):
    df1 = df.dropna(axis=0, how='any')
    m = df1.iloc[:, i].values[0]
    if type(m) is str:
        if len(df1.iloc[:, i].unique()) == 1:
            list_name = df1.columns.values.tolist()
            df.drop(columns=list_name[i], axis=1, inplace=True)
        else:
            pass
    else:
        pass

# 检查是否有空值
# 做填充：分类空值填其他，连续值用平均值填充
# 脏数据要在程序之外做处理
for i in range(df.shape[1]):
    n = df.iloc[:, i].values[3]
    # 分类变量，用其他填充空值
    if type(n) is str:
        list_name = df.columns.values.tolist()
        df[list_name[i]] = df[list_name[i]].astype('str')
        df.replace('nan', '其他', inplace=True)
    else:
        list_name = df.columns.values.tolist()
        df[list_name[i]] = df[list_name[i]].astype('float')
        imp = Imputer(missing_values='NaN', strategy='median', axis=0)
        X = df.iloc[:, i].values
        Y = X.reshape(-1, 1)
        imp.fit(Y)
        df.iloc[:, i] = imp.transform(Y)



# 组间差异比较
# 卡方检验
# 要先进行统计


a = df.iloc[:, 0].values
a = a.reshape(-1, 1)
b = df.iloc[:, 1].values
c = df.iloc[:, 2].values
c = c.reshape(-1, 1)
stats.ttest_ind(a, b, equal_var=True, nan_policy='omit')
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy.stats import chi2_contingency

kf_datad = np.array([a, c])
kf = chi2_contingency(kf_datad)

model1 = SelectKBest(chi2, k=2)  # 选择k个最佳特征
model1.fit_transform(c, a)  # iris.data是特征数据，iris.target是标签数据，该函数可以选择出k个特征

stats.pointbiserialr(data_b, data_p)
stats.ttest_ind(data_b, data_p, equal_var=True, nan_policy='omit')

stats.mannwhitneyu(data_b, data_p)
stats.ranksums(data_b, data_p)
stats.wilcoxon(data_b, data_p, zero_method='wilcox', correction=False)


##检验是否正态
def norm_test(data):
    t, p = stats.shapiro(data)
    # print(t,p)
    if p >= 0.05:
        return True
    else:
        return False


if norm_test(data_b) and norm_test(data_p):
    print('yes')
    t, p = stats.ttest_rel(list(data_b), list(data_p))
else:
    print('no')
    t, p = stats.wilcoxon(list(data_b), list(data_p), zero_method='wilcox', correction=False)


