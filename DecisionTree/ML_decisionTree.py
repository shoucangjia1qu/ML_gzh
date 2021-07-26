# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 22:40:19 2021

@author: ecupl
"""


import numpy as np
from utils import *
from base import BaseTree

# from sklearn.tree import DecisionTreeClassifier


# 分类树
class DecisionTreeClassifier(BaseTree):
    def __init__(self, criterion, minloss):
        """
        分类树的方法类

        Parameters
        ----------
        criterion : String
            分类树的准则，可选"entropy", "gini"，默认"entropy".
        minloss : float
            树停止生长的最小误差，分类树和回归树以及不同的准则会有不同的损失计算函数.

        Returns
        -------
        None.

        """
        super(DecisionTreeClassifier, self).__init__(criterion, minloss)
    
    
    def __search_feature_entropy(self, X, y, column_indicator=None):
        # 遍历特征，找到最大信息增益的特征
        n_samples, n_features = X.shape
        best_entropy_gainratio = 0
        best_feature_index = None
        best_feature_value = None
        for idx in range(n_features):
            if self.is_categorical.__len__() > 0 and self.is_categorical[column_indicator[idx]]==False:
                # 连续特征求信息增益率
                entropy_gainratio, feature_value = ent_gainratio_continuous(X[:, idx], y)
            else:
                # 分类特征求信息增益率
                entropy_gainratio, feature_value = ent_gainratio_categorical(X[:, idx], y)
            if entropy_gainratio > best_entropy_gainratio:
                best_entropy_gainratio = entropy_gainratio
                best_feature_index = idx
                best_feature_value = feature_value
        return column_indicator[best_feature_index], best_feature_value
    
    
    def __split_dataset_entropy(self, X, y, best_feature, feature_thres, column_indicator):
        best_feature_idx = column_indicator.index(best_feature)
        newx, newy, newcolindicator = np.copy(X), np.copy(y), list(np.copy(column_indicator))
        if self.is_categorical.__len__() > 0 and self.is_categorical[best_feature]==False:
            # 连续特征拆解数据集
            if feature_thres[1] == 'maximum':
                newx = newx[np.nonzero(X[:, best_feature_idx] <= feature_thres[0])[0], :]
                newy = newy[np.nonzero(X[:, best_feature_idx] <= feature_thres[0])[0]]
            elif feature_thres[1] == 'minimum':
                newx = newx[np.nonzero(X[:, best_feature_idx] > feature_thres[0])[0], :]
                newy = newy[np.nonzero(X[:, best_feature_idx] > feature_thres[0])[0]]
            else:
                pass
        else:
            # 分类特征拆解数据集
            newx = np.delete(newx[np.nonzero(X[:,best_feature_idx]==feature_thres)[0],:], best_feature_idx, axis=1)
            newy = newy[np.nonzero(X[:,best_feature_idx]==feature_thres)[0]]
            newcolindicator.remove(best_feature)
        return newx, newy, newcolindicator

    
    def fit(self, X, y, is_categorical:list=[]):
        # 赋值训练要用到的方法
        self._calculate_loss = calculate_entropy
        self._calculate_nodevalue_method = calculate_yproba
        self._search_feature_method = self.__search_feature_entropy
        self._split_dataset_method = self.__split_dataset_entropy
        # 其他参数
        self.is_categorical = is_categorical
        # 构建树的参数
        self.tree = self._build_tree(X, y, column_indicator=list(range(X.shape[1])))
        


def print_tree(tree):
    columns = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感', '密度', '含糖率']
    print(f"========第{tree.depth}层========")
    print(f"该树的结点类型为{tree.value}")
    if tree.childNode is None:
        return
    print(f"最优划分特征是{columns[tree.feature_i]}，划分值有{tree.feature_thres}")
    for i, node in tree.childNode.items():
        print(f"\n当子结点的属性值为{i}时：")
        print_tree(node)



if __name__ == "__main__":
    X = np.array([[0,0,0,0,0,1,1,1,1,1,2,2,2,2,2],[0,0,1,1,0,0,0,1,0,0,0,0,1,1,0],[0,0,0,1,0,0,0,1,1,1,1,1,0,0,0],[0,1,1,0,0,0,1,1,2,2,2,1,1,2,0]]).T
    y = np.array([0,0,1,1,0,0,0,1,1,1,1,1,1,1,0])
    # ID3分类树
    tree_id3_classifier = DecisionTreeClassifier(criterion="entropy", minloss=0.0001)
    tree_id3_classifier.fit(X, y)
    tree_id3_classifier.tree
    print_tree(tree_id3_classifier.tree)
    
    # C4.5对西瓜集数据分类
    tree_c45_classifier = DecisionTreeClassifier(criterion="entropy", minloss=0.0001)
    tree_c45_classifier.fit(Xdata, Ylabel, [True, True, True, True, True, True, False, False])
    tree_c45_classifier.tree
    print_tree(tree_c45_classifier.tree)



    