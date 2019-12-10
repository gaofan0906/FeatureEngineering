# 定义数据类型
# 人工的去看，哪些是离散（str），哪些是连续（float）
# 手动处理脏数据
df['列名']=df['列名'].astype("类型")

# 删除填充率低的数列
# 计算填充率，并删除填充率低于15%的列
def fill(data):
    '''data=dataframe'''

    # 计算非空数值的个数，添加一行
    num=data.count()
    df1=data.append(num,ignore_index=True)
    # 计算填充率=（非空值/总行数）*100%
    sum1=df1.iloc[-1]/data.shape[0]
    df2=data.append(sum1,ignore_index=True)
    # sum2=(df1.iloc[-1]/df.shape[0]).apply(lambda x: format(x, '.2%'))
    # df3=df.append(sum2,ignore_index=True)

    # 删除填充率低于15%的列
    # 判断是否小于15%，拿到列名删除
    drop_names=[]
    for i in df2.columns:
        if not df2[i].iloc[-1]>.85:
            drop_names.append(i)

    df3=data.drop(columns=drop_names)
    return df3

df_new=fill(data=df)

# 补差
# 检查是否有空值
# 做填充：分类空值填其他，连续值用平均值填充
# 脏数据要在程序之外做处理
from sklearn.preprocessing import Imputer
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


# 归一化
# （0，1）标准化
# 找大小的方法直接用np.max()和np.min()
def MaxMinNormalization(x,Max,Min):
    x = (x - Min) / (Max - Min);
    return x
# Z-score标准化
# mu（即均值）用np.average()，sigma（即标准差）用np.std()
def Z_ScoreNormalization(x,mu,sigma):
    x = (x - mu) / sigma;
    return x
# Sigmoid函数
def sigmoid(X,useStatus):
    if useStatus:
        return 1.0 / (1 + np.exp(-float(X)));
    else:
        return float(X)
# PCA