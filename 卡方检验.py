# -*- coding: UTF-8 -*-
# python 3.5.0
# 卡方计算
__author__ = 'HZC'

import math
import sqlalchemy
import numpy as np
import pandas as pd


class CHISQUARE:
    def __init__(self, d):
        self.engine = sqlalchemy.create_engine("mssql+pymssql://%s:%s@%s/%s" % (d['user'], d['pwd'], d['ins'], d['db']))

    def get_df_from_query(self, sql):
        df = pd.read_sql_query(sql, self.engine)
        return df

    def get_variance(self, df):
        row_count = df.shape[0] - 1
        col_count = df.shape[1] - 1
        v = (row_count - 1) * (col_count - 1)
        return v

    # 转为矩阵求卡方距离
    def get_chi_square_value(self, df1, df2):
        df1 = df1.drop(['col_total'])
        df2 = df2.drop(['col_total'])
        del df1['row_total']
        del df2['row_total']
        mtr1 = df1.astype(int).as_matrix()
        mtr2 = df2.astype(int).as_matrix()
        mtr = ((mtr1 - mtr2) ** 2) / mtr2
        return mtr.sum()

    # 分类频数
    def get_classification(self, table_name, col_result, col_pred):
        sql = "select %s,%s from %s" % (col_result, col_pred, table_name)
        df = self.get_df_from_query(sql)
        df = df.groupby([col_result, col_pred]).agg({col_result: ['count']})
        df = df.reset_index()
        df.columns = [col_result, col_pred, 'count']
        df = pd.pivot_table(df, values='count', index=col_result, columns=col_pred).reset_index()
        df['row_total'] = df.sum(axis=1)
        df.set_index(col_result, inplace=True)
        df.loc['ratio(%)'] = df.loc[0] * 100 / df.loc[1]
        print("==========================================================")
        print("原始数据粗分类：（百分比相近的可划分为同一类）")
        print("==========================================================")
        print(df.astype(int))
        df = df.drop(['ratio(%)'])
        df.loc['col_total'] = df.sum(axis=0)
        print("==========================================================")
        print("分类频数汇总：（实际值）")
        print("==========================================================")
        print(df.astype(int))
        df2 = df.copy()
        total = df2[['row_total']].loc[['col_total']].values[0][0]
        for col in df2:
            df2[col] = df2[[col]].loc[['col_total']].values[0][0] * df2['row_total'] / total
        df2 = df2.drop(['col_total'])
        df2.loc['col_total'] = df2.sum(axis=0)
        print("==========================================================")
        print("期望频数分布：（理论推算值）")
        print("与上表差距越大，两表的独立性越低、依存度越高，粗分类效果越好")
        print("==========================================================")
        print(df2.astype(int))
        print("==========================================================")
        x = self.get_chi_square_value(df, df2)  # 顺序：(实际df,推算df)
        v = self.get_variance(df2)  # v=（行数-1）（列数-1）
        print("卡方值：χ2 = %s" % x)
        print("自由度：v = %s" % v)
        print("==========================================================")


if __name__ == "__main__":
    conn = {'user': '用户名', 'pwd': '密码', 'ins': '实例', 'db': '数据库'}
    cs = CHISQUARE(conn)
    cs.get_classification("V_ClientInfoAll", "是否回款", "婚姻状况")
# cs.get_classification(表或视图,回归只/判断值,"分类元素")