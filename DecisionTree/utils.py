# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 22:42:04 2021

@author: ecupl
"""


import numpy as np



# 计算信息熵
def calculate_entropy(y):
    entropy = 0
    for value in set(y):
        entropy+=-(sum(y==value)/len(y))*np.log2(sum(y==value)/len(y) + 1.0e-5)
    return entropy


# 计算分类的概率
def calculate_yproba(y):
    ydict = {}
    ySet = np.unique(y)
    for i in ySet:
        ydict[i] = round(sum(y==i)/len(y), 3)
    return ydict
    

# 拆分数据集函数
