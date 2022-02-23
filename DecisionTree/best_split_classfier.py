# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 21:46:54 2022

@author: ecupl
"""


import numpy as np
from utils import *


# 计算信息增益率——分类变量
def ent_cutoff_categorical(xarray, yarray):
    """
    分类变量根据信息增益率计算最优切分点
    """
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
def ent_cutoff_continuous(xarray, yarray):
    """
    连续变量根据信息增益率计算最优切分点
    """
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


# 计算同一个特征中最优划分点的条件基尼指数__分类变量
def gini_cutoff_categorical(xarray, yarray):
    """
    分类变量根据GINI指数计算最优切分点
    """
    n_samples = len(yarray)
    xSet = np.unique(xarray)
    best_gini = 1.0
    best_feature_value = None
    for xi in xSet:
        # 计算xi为切分点时的基尼指数
        y_left = yarray[xarray==xi]
        gini_left = calculate_gini(y_left)
        # print(y_left.__len__(), gini_left)
        y_right = yarray[xarray!=xi]
        gini_right = calculate_gini(y_right)
        # print(y_right.__len__(), gini_right)
        gini = (len(y_left)*gini_left + len(y_right)*gini_right)/n_samples
        # print(xi, n_samples, gini)
        if gini < best_gini:
            best_gini = gini
            best_feature_value = xi
    return best_gini, [(best_feature_value, 'left'), (best_feature_value, 'right')]



# 计算同一个特征中最优划分点的条件基尼指数__连续变量
def gini_cutoff_continuous(xarray, yarray):
    """
    连续变量根据GINI指数计算最优切分点
    """
    n_samples = len(yarray)
    xSet = np.unique(xarray)
    best_gini = 1.0
    best_feature_value = None
    for xi in xSet:
        # 计算xi为切分点时的基尼指数
        y_left = yarray[xarray<=xi]
        gini_left = calculate_gini(y_left)
        y_right = yarray[xarray>xi]
        gini_right = calculate_gini(y_right)
        gini = (len(y_left)*gini_left + len(y_right)*gini_right)/n_samples
        if gini < best_gini:
            best_gini = gini
            best_feature_value = xi
    return best_gini, [(best_feature_value, 'left'), (best_feature_value, 'right')]
