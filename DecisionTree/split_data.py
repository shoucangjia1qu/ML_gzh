# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 21:46:54 2022

@author: ecupl
"""


import numpy as np



# gini分类树和mse的回归树都是用左右子树的划分方法
def split_dataset_cart(X, y, best_feature, feature_thres, is_categorical, column_indicator):
    best_feature_idx = column_indicator.index(best_feature)
    newx, newy, newcolindicator = np.copy(X), np.copy(y), list(np.copy(column_indicator))
    if (is_categorical.__len__() > 0 and is_categorical[best_feature]==True):
        # 分类特征拆解数据集
        if feature_thres[1] == 'left':
            newx = newx[np.nonzero(X[:, best_feature_idx] == feature_thres[0])[0], :]
            newy = newy[np.nonzero(X[:, best_feature_idx] == feature_thres[0])[0]]
        elif feature_thres[1] == 'right':
            newx = newx[np.nonzero(X[:, best_feature_idx] != feature_thres[0])[0], :]
            newy = newy[np.nonzero(X[:, best_feature_idx] != feature_thres[0])[0]]
        else:
            pass
    else:
        # 连续特征拆解数据集
        if feature_thres[1] == 'left':
            newx = newx[np.nonzero(X[:, best_feature_idx] <= feature_thres[0])[0], :]
            newy = newy[np.nonzero(X[:, best_feature_idx] <= feature_thres[0])[0]]
        elif feature_thres[1] == 'right':
            newx = newx[np.nonzero(X[:, best_feature_idx] > feature_thres[0])[0], :]
            newy = newy[np.nonzero(X[:, best_feature_idx] > feature_thres[0])[0]]
        else:
            pass
    return newx, newy, newcolindicator






SPLIT_DATA = {
    'mse' : split_dataset_cart,
    'gini' : split_dataset_cart,
    
    }

