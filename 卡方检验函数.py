import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from scipy.stats import chi2_contingency
from scipy import stats
import statsmodels.api as sm

data_path = './data0930.csv'
df = pd.read_csv(data_path)
df.drop(columns='Unnamed: 0', inplace=True)

# 分类的列拿出来
# 如何区分离散变量和连续变量
k2_2 = []
k2_3 = []
k2_4 = []
for i in range(1, df.shape[1]):
    if type(df.iloc[:, i][1]) == type(df.iloc[:, 0][1]):
        if df.iloc[:, i].max() == 1:
            count2_00 = 0
            count2_01 = 0
            count2_10 = 0
            count2_11 = 0
            l1 = zip(list(df.iloc[:, 0]), list(df.iloc[:, i]))
            for l in l1:
                if l == (0, 0):
                    count2_00 += 1
                elif l == (0, 1):
                    count2_01 += 1
                elif l == (1, 0):
                    count2_10 += 1
                elif l == (1, 1):
                    count2_11 += 1
            # 计数中不能有0值
            a1 = [count2_10,
                  count2_11]
            b1 = [count2_00,
                  count2_01]
            kf_datad2 = np.array([a1, b1])
            # 卡方检验
            k2_2.append(chi2_contingency(kf_datad2))
        elif df.iloc[:, i].max() == 2:
            count3_00 = 0
            count3_01 = 0
            count3_02 = 0
            count3_10 = 0
            count3_11 = 0
            count3_12 = 0
            l2 = zip(list(df.iloc[:, 0]), list(df.iloc[:, i]))
            for j in l2:
                if j == (0, 0):
                    count3_00 += 1
                elif j == (0, 1):
                    count3_01 += 1
                elif j == (0, 2):
                    count3_02 += 1
                elif j == (1, 0):
                    count3_10 += 1
                elif j == (1, 1):
                    count3_11 += 1
                elif j == (1, 2):
                    count3_12 += 1
            # 计数中不能有0值
            a2 = [count3_10,
                  count3_11, count3_12]
            b2 = [count3_00,
                  count3_01, count3_02]
            kf_datad3 = np.array([a2, b2])
            # 卡方检验
            k2_3.append(chi2_contingency(kf_datad3))
        elif df.iloc[:, i].max() == 3:
            count4_00 = 0
            count4_01 = 0
            count4_02 = 0
            count4_03 = 0
            count4_10 = 0
            count4_11 = 0
            count4_12 = 0
            count4_13 = 0
            l3 = zip(list(df.iloc[:, 0]), list(df.iloc[:, i]))
            for k in l3:
                if k == (0, 0):
                    count4_00 += 1
                elif k == (0, 1):
                    count4_01 += 1
                elif k == (0, 2):
                    count4_02 += 1
                elif k == (0, 3):
                    count4_03 += 1
                elif k == (1, 0):
                    count4_10 += 1
                elif k == (1, 1):
                    count4_11 += 1
                elif k == (1, 2):
                    count4_12 += 1
                elif k == (1, 3):
                    count4_13 += 1
            # 计数中不能有0值
            a3 = [count4_10,
                  count4_11, count4_12]
            b3 = [count4_00,
                  count4_01, count4_02]
            kf_datad4 = np.array([a3, b3])
            # 卡方检验
            k2_4.append(chi2_contingency(kf_datad4))
        else:
            pass


zk=[]
t=[]
for i in range(1, df.shape[1]):
    if not type(df.iloc[:, i][1]) == type(df.iloc[:, 0][1]):
        # _,p=stats.shapiro(df.iloc[:, i])
        # if p<0.05:
        grouped=df.iloc[:, i].groupby(df.iloc[:, 0])
        c = []
        d = []
        for group_name, group_data in grouped:
            if group_name == 0:
                for i in group_data:
                    c.append(i)
            else:
                for i in group_data:
                    d.append(i)
        zk.append(stats.ranksums(c, d))
        # else:
        #     grouped = df.iloc[:, i].groupby(df.iloc[:, 0])
        #     c = []
        #     d = []
        #     for group_name, group_data in grouped:
        #         if group_name == 0:
        #             for i in group_data:
        #                 c.append(i)
        #         else:
        #             for i in group_data:
        #                 d.append(i)
        #     t.append(stats.ttest_rel(c, d))
# 单因素回归
df["性别"] = df["性别"].astype(int)
list_name = df.columns.values.tolist()
name=list_name[1:]
model = sm.Logit(endog=df["性别"], exog=df[name]).fit()
print(model.summary())

# 多因素回归
model = sm.OLS(endog=df["性别"], exog=df[name]).fit()
print(model.summary())
# P<0.05则认为自变量具有统计学意义