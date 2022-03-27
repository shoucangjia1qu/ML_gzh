# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 21:46:54 2022

@author: ecupl
"""


import numpy as np
from tools.utils import *





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
    return best_gain_entropy_ratio, [(best_feature_value, 'left'), (best_feature_value, 'right')]







# 计算同一个特征中最优划分点的条件基尼指数
def _cutoff_gini(xarray, yarray, min_samples_leaf, category_flag):
    """
    连续变量根据GINI指数计算最优切分点
    """
    n_samples = len(yarray)
    xSet = np.unique(xarray)
    best_gini = 1.0
    best_feature_value = None
    for xi in xSet:
        # 计算xi为切分点时的基尼指数
        if category_flag:
            y_left = yarray[xarray==xi]
            y_right = yarray[xarray!=xi]
        else:
            y_left = yarray[xarray<=xi]
            y_right = yarray[xarray>xi]
        # 当子树不满足最小样本数，跳过
        if (len(y_left) < min_samples_leaf) or (len(y_right) < min_samples_leaf):
            continue
        gini_left = calculate_gini(y_left)
        gini_right = calculate_gini(y_right)
        gini = (len(y_left)*gini_left + len(y_right)*gini_right)/n_samples
        if gini < best_gini:
            best_gini = gini
            best_feature_value = xi
    return best_gini, [(best_feature_value, 'left'), (best_feature_value, 'right')]


def search_best_split_feature_gini(X, y, min_samples_leaf, is_categorical, column_indicator):
    # 遍历特征，根据GINI指数最小化准则找到最优划分特征和切分点
    n_samples, n_features = X.shape
    best_gini = 1
    bestsplit_feature_index = None
    bestsplit_value = None
    
    for idx in range(n_features):
        # 分类变量标识
        cat_flag = False
        if (is_categorical.__len__() > 0 and is_categorical[column_indicator[idx]]==True):
            cat_flag = True
        feature_gini, split_value = _cutoff_gini(X[:, idx], y, min_samples_leaf, cat_flag)
        if feature_gini < best_gini:
            best_gini = feature_gini
            bestsplit_feature_index = idx
            bestsplit_value = split_value
    return column_indicator[bestsplit_feature_index] if bestsplit_feature_index is not None else None, bestsplit_value






# 计算同一个特征中最优划分点的条件MSE
def _cutoff_mse(xarray, yarray, min_samples_leaf, category_flag):
    """
    回归树中，单个特征根据MSE计算最优切分点
    """
    xSet = np.unique(xarray)
    best_mse = np.inf
    best_feature_value = None
    for xi in xSet:
        # 区分连续变量和分类变量计算左右子树
        if category_flag:
            y_left = yarray[xarray==xi]
            y_right = yarray[xarray!=xi]
        else:
            y_left = yarray[xarray<=xi]
            y_right = yarray[xarray>xi]
        # 当子树不满足最小样本数，跳过
        if (len(y_left) < min_samples_leaf) or (len(y_right) < min_samples_leaf):
            continue
        # 计算xi为切分点时的均方误差MSE
        mse_left = calculate_mse(y_left)
        mse_right = calculate_mse(y_right)
        mse = mse_left + mse_right
        if mse < best_mse:
            best_mse = mse
            best_feature_value = xi
    return best_mse, [(best_feature_value, 'left'), (best_feature_value, 'right')]


def search_best_split_feature_mse(X, y, min_samples_leaf, is_categorical, column_indicator):
    # 遍历特征，根据MSE最小化准则找到最优划分特征和切分点
    n_samples, n_features = X.shape
    best_mse = np.inf
    bestsplit_feature_index = None
    bestsplit_value = None
    # 遍历每个特征，找到最优分割点
    for idx in range(n_features):
        # 分类变量标识
        cat_flag = False
        if (is_categorical.__len__() > 0 and is_categorical[column_indicator[idx]]==True):
            cat_flag = True
        feature_mse, split_value = _cutoff_mse(X[:, idx], y, min_samples_leaf, cat_flag)
        if feature_mse < best_mse:
            best_mse = feature_mse
            bestsplit_feature_index = idx
            bestsplit_value = split_value
    # print(column_indicator[bestsplit_feature_index], bestsplit_value)
    return column_indicator[bestsplit_feature_index] if bestsplit_feature_index is not None else None, bestsplit_value




SEARCH_FEATURE = {
    'mse' : search_best_split_feature_mse,
    'gini' : search_best_split_feature_gini
    
    
    }
