# 差异性检验
import pandas as pd
from scipy.stats import shapiro,mannwhitneyu,ttest_ind,chi2_contingency
from numpy import array
import statsmodels.api as sm
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.model_selection import GridSearchCV

df1=pd.read_csv('./DTdata1025.csv')
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

# 数据描述
df1.describe()
df2 = df1[df1['group'] == 0]
df3 = df1[df1['group'] == 1]
df2.describe()
df3.describe()


pd.crosstab(df2['group'], df2['sex_1'], rownames=['group'])
pd.crosstab(df2['group'], df2['sex_2'], rownames=['group'])

pd.crosstab(df2['group'], df2['BMI_group_1.0'], rownames=['group'])
pd.crosstab(df2['group'], df2['BMI_group_1.8646288209606987'], rownames=['group'])
pd.crosstab(df2['group'], df2['BMI_group_2.0'], rownames=['group'])
pd.crosstab(df2['group'], df2['BMI_group_3.0'], rownames=['group'])
pd.crosstab(df2['group'], df2['yaowei_group_1.0'], rownames=['group'])
pd.crosstab(df2['group'], df2['yaowei_group_1.864321608040201'], rownames=['group'])
pd.crosstab(df2['group'], df2['yaowei_group_2.0'], rownames=['group'])

pd.crosstab(df2['group'], df2['smoking_1'], rownames=['group'])
pd.crosstab(df2['group'], df2['smoking_2'], rownames=['group'])
pd.crosstab(df2['group'], df2['drink_1'], rownames=['group'])
pd.crosstab(df2['group'], df2['drink_2'], rownames=['group'])


pd.crosstab(df3['group'], df3['sex_1'], rownames=['group'])
pd.crosstab(df3['group'], df3['sex_2'], rownames=['group'])

pd.crosstab(df3['group'], df3['BMI_group_1.0'], rownames=['group'])
pd.crosstab(df3['group'], df3['BMI_group_1.8646288209606987'], rownames=['group'])
pd.crosstab(df3['group'], df3['BMI_group_2.0'], rownames=['group'])
pd.crosstab(df3['group'], df3['BMI_group_3.0'], rownames=['group'])
pd.crosstab(df3['group'], df3['yaowei_group_1.0'], rownames=['group'])
pd.crosstab(df3['group'], df3['yaowei_group_1.864321608040201'], rownames=['group'])
pd.crosstab(df3['group'], df3['yaowei_group_2.0'], rownames=['group'])

pd.crosstab(df3['group'], df3['smoking_1'], rownames=['group'])
pd.crosstab(df3['group'], df3['smoking_2'], rownames=['group'])
pd.crosstab(df3['group'], df3['drink_1'], rownames=['group'])
pd.crosstab(df3['group'], df3['drink_2'], rownames=['group'])

# 正态分布检验
# 秩和检验，t检验
name=['follow_up',
 'bingcheng',
 'age_fordiagnosis',
 'shuzhangya',
 'shousuoya',
 'ALB',
 'ALT',
 'LDL_C',
 'TG',
 'HDL_C',
 'Crea',
 'Bun',
 'UA',
 'Glu',
 'HbA1c',
 'AST',
 'Hb',
 'FT4',
 'FT3',
 'TC',
 'Total_bilirubin']
def get_class(x,Y):
    tmp = pd.DataFrame(list(zip(x, Y)))
    group = list(tmp.groupby(1))
    group_0=group[0]
    num_0=array(group_0[1])
    ar_0=num_0[:,0]

    group_1 = group[1]
    num_1 = array(group_1[1])
    ar_1=num_1[:,0]
    return ar_0,ar_1
res=[]
for i in name:
    _,p=shapiro(df1[i])
    if p>0.05:
        a,b=get_class(df1[i],df1['group'])
        res.append(ttest_ind(a, b))
    else:
        a, b = get_class(df1[i], df1['group'])
        res.append(mannwhitneyu(a, b))

# 卡方检验
chi2_contingency([[174,347],[65,103]])
chi2_contingency([[194,2,220,105],[45,0,82,41]])
chi2_contingency([[68,76,377],[13,16,139]])
chi2_contingency([[384,137],[125,43]])
chi2_contingency([[453,68],[141,27]])

# 单因素回归
for_name1=[
 'follow_up',
 'bingcheng',
 'age_fordiagnosis',
 'shuzhangya',
 'shousuoya',
 'ALB',
 'ALT',
 'LDL_C',
 'TG',
 'HDL_C',
 'Crea',
 'Bun',
 'UA',
 'Glu',
 'HbA1c',
 'AST',
 'Hb',
 'FT4',
 'FT3',
 'TC',
 'Total_bilirubin',
 'sex_2',
 'BMI_group_1.0',
 'BMI_group_2.0',
 'BMI_group_3.0',
 'yaowei_group_1.0',
 'yaowei_group_2.0',
 'smoking_2',
 'drink_2']
for i in for_name1:
    model = sm.Logit(endog=df1['group'], exog=df1[['intercept',i]]).fit()
    print(model.summary())

# 多因素回归
df1['intercept']=1
for_name2=['intercept',
 'follow_up',
 'bingcheng',
 'age_fordiagnosis',
 'shuzhangya',
 'shousuoya',
 'ALB',
 'ALT',
 'LDL_C',
 'TG',
 'HDL_C',
 'Crea',
 'Bun',
 'UA',
 'Glu',
 'HbA1c',
 'AST',
 'Hb',
 'FT4',
 'FT3',
 'TC',
 'Total_bilirubin',
 'sex_2',
 'BMI_group_1.0',
 'BMI_group_2.0',
 'yaowei_group_1.0',
 'yaowei_group_2.0',
 'smoking_2',
 'drink_2']
model = sm.Logit(endog=df1['group'], exog=df1[for_name2]).fit()
print(model.summary())

# 逻辑回归

X = df1.drop(columns=['group','intercept'])
Y = df1['group']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


# 3.标准化特征值
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# gridsearch寻找最优参数
# 用GridSearchCV寻找最优参数（字典）
param = {'penalty':['l1','l2'],'C':[1.0,1e5,1e9],'class_weight':[None,{0:0.2, 1:0.8},{0:0.3, 1:0.7}]}
grid = GridSearchCV(linear_model.LogisticRegression(),param_grid=param,cv=6)
grid.fit(X_train,Y_train)
print('最优分类器:',grid.best_params_,'最优分数:', grid.best_score_)  # 得到最优的参数和分值

best_model=grid.best_estimator_

# 交叉验证
scores = cross_val_score(best_model, X, Y, cv=10)
print(scores)
# 5. 预测
y_pred=best_model.predict(X_test)

# 评价指标
c=metrics.classification_report(Y_test, y_pred)

# 画ROC曲线
###通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
y_score = best_model.decision_function(X_test)
fpr, tpr, threshold = roc_curve(Y_test, y_score)  ###计算真正率和假正率
roc_auc = auc(fpr, tpr)  ###计算auc的值
plt.figure()
lw = 2
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig('./LRauc.png')
plt.show()


# 画学习曲线
train_sizes, train_scores, test_scores = learning_curve(estimator=best_model, X=X_train, y=Y_train,
                                                        train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)
# 统计结果
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
# 绘制效果
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='test accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.6, 1.0])
plt.savefig('./LRlearningr.png')
plt.show()



