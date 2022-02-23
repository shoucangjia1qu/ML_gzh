# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 21:46:54 2022

@author: ecupl
"""


import numpy as np
from utils import calculate_mse



# 计算同一个特征中最优划分点的MSE__分类变量
def mse_cutoff_categorical(xarray, yarray):
    """
    回归树中，分类变量根据MSE计算最优切分点
    """
    xSet = np.unique(xarray)
    best_mse = np.inf
    best_feature_value = None
    for xi in xSet:
        # 计算xi为切分点时的均方误差MSE
        y_left = yarray[xarray==xi]
        mse_left = calculate_mse(y_left)
        # print(y_left.__len__(), gini_left)
        y_right = yarray[xarray!=xi]
        mse_right = calculate_mse(y_right)
        # print(y_right.__len__(), gini_right)  
        # 计算总的均方误差MSE
        mse = mse_left + mse_right
        # print(xi, n_samples, gini)
        if mse < best_mse:
            best_mse = mse
            best_feature_value = xi
    return best_mse, [(best_feature_value, 'left'), (best_feature_value, 'right')]



# 计算同一个特征中最优划分点的条件MSE__连续变量
def mse_cutoff_continuous(xarray, yarray):
    """
    回归树中，连续变量根据MSE计算最优切分点
    """
    xSet = np.unique(xarray)
    best_mse = np.inf
    best_feature_value = None
    for xi in xSet:
        # 计算xi为切分点时的均方误差MSE
        y_left = yarray[xarray<=xi]
        mse_left = calculate_mse(y_left)
        y_right = yarray[xarray>xi]
        mse_right = calculate_mse(y_right)
        mse = mse_left + mse_right
        if mse < best_mse:
            best_mse = mse
            best_feature_value = xi
    return best_mse, [(best_feature_value, 'left'), (best_feature_value, 'right')]
