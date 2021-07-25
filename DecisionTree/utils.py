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
    

# 计算信息增益率——分类变量
def ent_gainratio_categorical(xarray, yarray):
    root_entropy = calculate_entropy(yarray)
    xSet = np.unique(xarray)
    Di = [sum(xarray == value) / len(xarray) for value in xSet]
    Child_entropy = [calculate_entropy(yarray[np.nonzero(xarray == value)[0]]) for value in xSet]
    # ID3，计算信息增益
    gain_entropy = root_entropy - np.dot(Di, Child_entropy)
    # C4.5，计算信息增益率
    iv = calculate_entropy(xarray)
    gain_entropy_ratio = gain_entropy / iv
    return gain_entropy_ratio, list(xSet)


# 计算信息增益率——连续变量
def ent_gainratio_continuous(xarray, yarray):
    root_entropy = calculate_entropy(yarray)
    xSet = np.unique(xarray)
    # best_gain_entropy = 0
    best_gain_entropy_ratio = 0
    best_feature_value = None
    feature_value_list = [(xSet[i] + xSet[i + 1]) / 2 for i in range(len(xSet) - 1)]
    for value in feature_value_list:
        Di = [sum(xarray <= value) / len(xarray) , sum(xarray > value) / len(xarray)]
        Child_entropy = [calculate_entropy(yarray[np.nonzero(xarray < value)[0]]), calculate_entropy(yarray[np.nonzero(xarray > value)[0]])]
        # ID3，计算信息增益
        gain_entropy = root_entropy - np.dot(Di, Child_entropy)
        # C4.5，计算信息增益率
        iv = calculate_entropy(xarray)
        gain_entropy_ratio = gain_entropy / iv
        if gain_entropy_ratio > best_gain_entropy_ratio:
            # best_gain_entropy = gain_entropy
            best_gain_entropy_ratio = gain_entropy_ratio
            best_feature_value = value
    return best_gain_entropy_ratio, [(best_feature_value, 'maximum'), (best_feature_value, 'minimum')]



