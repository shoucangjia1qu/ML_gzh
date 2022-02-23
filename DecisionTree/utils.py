# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 22:42:04 2021

@author: ecupl
"""


import numpy as np





# 计算分类的概率
def calculate_yproba(y):
    ydict = {}
    ySet = np.unique(y)
    for i in ySet:
        ydict[i] = round(sum(y==i)/len(y), 3)
    return ydict



# 计算结点的均值
calculate_mean = lambda x: np.mean(x)



# 计算信息熵
def calculate_entropy(y):
    entropy = 0
    for value in set(y):
        entropy+=-(sum(y==value)/len(y))*np.log2(sum(y==value)/len(y) + 1.0e-5)
    return entropy



# 计算GINI指数
def calculate_gini(y):
    if len(y)==0:
        return 0
    ycount = len(y)
    yset, yicount = np.unique(y, return_counts=True)
    # 二分类用简化算法
    if len(yset)==2:
        gini = 2*yicount[0]*(ycount-yicount[0])/ycount**2
    else:
        gini = ycount**2
        for i in yicount:
            gini -= i**2
        gini = gini/ycount**2
    return gini


# 计算均方误差MSE
def calculate_mse(y):
    '''计算均方误差Mean Squared Error'''
    ypred = np.mean(y)
    mse = np.sum(np.power(y - ypred, 2))
    return mse



