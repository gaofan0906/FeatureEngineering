import pandas as pd
from scipy import stats
import numpy as np

data_path = ('./0903datanew.csv')
df = pd.read_csv(data_path)

# 连续值补差
from sklearn.preprocessing import Imputer
import numpy as np

# 导入要进行缺失值处理的数据文件，数据文件上面有展示
data = np.genfromtxt('input.csv', skip_header=True, delimiter=',')

imp = Imputer(missing_values='NaN', strategy='median', axis=0)
# 上面'NAN'表示无效值在数据文件中的标识是'NAN',strategy='mean'表示用全局平均值代替无效值，axis=0表示对列进行处理

imp.fit(data)
# 训练一个缺失值处理模型

outfile = imp.transform(data)

# 存储到本地
np.savetxt('outfile.csv', outfile, delimiter=',')


X=df.iloc[:,4].values
X=X.reshape(-1,1)

imp.fit(X)
# 训练一个缺失值处理模型

outfile = imp.transform(X)

print(stats.shapiro(outfile))
print(stats.kstest(outfile, 'norm'))

