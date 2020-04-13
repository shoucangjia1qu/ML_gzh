# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def coding(df, columns, method):
    """小型分类变量的编码处理
    Parameters:
    -----------
    df: pandas.DataFrame
        需要编码的df
    columns: list-like
        df中需要编码的特征名称
    method: string
        编码处理方式, "one-hot", "dummy", "effect"可选
    
    Returns:
    --------
    codingDF: pandas.DataFrame
        返回编码处理过的df
    """
    if method.lower() == "one-hot":
        codingDF = pd.get_dummies(df, columns=columns)
    elif method.lower() == "dummy":
        codingDF = pd.get_dummies(df, columns=columns, drop_first=True)
    elif method.lower() == "effect":
        dummyDF = pd.get_dummies(df, columns=columns, drop_first=True)
        codingDF = dummy2effect(dummyDF, columns)
    return codingDF
        
def dummy2effect(dummyDF, columns):
    """将经过虚拟编码的DF转变为效果编码的形式"""
    dummyColumns = dummyDF.columns.tolist()
    effectCols = [[newCol for newCol in dummyColumns if newCol.split("_")[0]==col] for col in columns ]
    effectDF = dummyDF.copy()
    for pairCols in effectCols:
        featureidx = [dummyColumns.index(c) for c in pairCols]
        sampleidx = np.nonzero(dummyDF[effectCols[0]].sum(axis=1).values == 0)[0]
        effectDF.iloc[sampleidx, featureidx] = -1.
    return effectDF
    
    
if __name__ == "__main__":
    df = pd.DataFrame({'CITY':['BJ','BJ','BJ','SH','SH','SH','SZ','SZ','SZ'], 
                   'RENT':[3999,4000,4001,3499,3500,3501,2999,3000,3001]})
    #分别进行编码
    onehot_df = coding(df, ["CITY"], "one-hot")
    dummy_df = coding(df, ["CITY"], "dummy")
    effect_df = coding(df, ["CITY"], "effect")
    #分别进行线性回归
    from sklearn import linear_model
    clf_onehot = linear_model.LinearRegression()
    clf_onehot.fit(onehot_df.drop("RENT", axis=1), onehot_df["RENT"])
    clf_dummy = linear_model.LinearRegression()
    clf_dummy.fit(dummy_df.drop("RENT", axis=1), dummy_df["RENT"])
    clf_effect = linear_model.LinearRegression()
    clf_effect.fit(effect_df.drop("RENT", axis=1), effect_df["RENT"])
    #三种编码训练的系数
    """
                CITY_BJ     CITY_SH     CITY_SZ       intercept
    one-hot     500.00      0.00        -500.00     3500.00
    dummy       -           500.00      -1000.00    4000.00
    effect      -           0.00        -500.00     3500.00
    """

    """
    我们可以发现，在one-hot编码中，截距代表了整体Y值（租金）的均值，系数代表了相应城市的租金均值与整体租金均值有多大差别。
    在虚拟编码中，截距代表了参照类Y值（租金）的均值，本例中的参照类是BJ，系数代表了相应城市的租金均值与参照类均值的差异。
    在效果编码中，截距代表了整体Y值（租金）的均值，各个系数表示了各个类别的均值与整体均值之间的差，此处BJ的均值当且仅当"CITY_SH=CITY_SZ=-1"时所取到的值。
    """
    

