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
        bestsplit_feature_index = None
        bestsplit_value = None
        for idx in range(n_features):
            if self.is_categorical.__len__() > 0 and self.is_categorical[column_indicator[idx]]==False:
                # 连续特征求信息增益率
                entropy_gainratio, feature_value = ent_gainratio_continuous(X[:, idx], y)
            else:
                # 分类特征求信息增益率
                entropy_gainratio, feature_value = ent_gainratio_categorical(X[:, idx], y)
            if entropy_gainratio > best_entropy_gainratio:
                best_entropy_gainratio = entropy_gainratio
                bestsplit_feature_index = idx
                bestsplit_value = feature_value
        return column_indicator[bestsplit_feature_index], bestsplit_value
    
    
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

    
    def __search_feature_gini(self, X, y, column_indicator=None):
        # 遍历特征，根据GINI指数最小化准则找到最优划分特征和切分点
        n_samples, n_features = X.shape
        best_gini = 1
        bestsplit_feature_index = None
        bestsplit_value = None
        for idx in range(n_features):
            if self.is_categorical.__len__() > 0 and self.is_categorical[column_indicator[idx]]==False:
                # 连续特征求GINI指数
                feature_gini, split_value = gini_cutoff_continuous(X[:, idx], y)
            else:
                # 分类特征求GINI指数
                feature_gini, split_value = gini_cutoff_categorical(X[:, idx], y)
            if feature_gini < best_gini:
                best_gini = feature_gini
                bestsplit_feature_index = idx
                bestsplit_value = split_value
        # print(column_indicator[bestsplit_feature_index], bestsplit_value)
        return column_indicator[bestsplit_feature_index], bestsplit_value
    
    
    def __split_dataset_gini(self, X, y, best_feature, feature_thres, column_indicator):
        best_feature_idx = column_indicator.index(best_feature)
        newx, newy, newcolindicator = np.copy(X), np.copy(y), list(np.copy(column_indicator))
        if self.is_categorical.__len__() > 0 and self.is_categorical[best_feature]==False:
            # 连续特征拆解数据集
            if feature_thres[1] == 'left':
                newx = newx[np.nonzero(X[:, best_feature_idx] <= feature_thres[0])[0], :]
                newy = newy[np.nonzero(X[:, best_feature_idx] <= feature_thres[0])[0]]
            elif feature_thres[1] == 'right':
                newx = newx[np.nonzero(X[:, best_feature_idx] > feature_thres[0])[0], :]
                newy = newy[np.nonzero(X[:, best_feature_idx] > feature_thres[0])[0]]
            else:
                pass
        else:
            # 分类特征拆解数据集
            if feature_thres[1] == 'left':
                newx = newx[np.nonzero(X[:, best_feature_idx] == feature_thres[0])[0], :]
                newy = newy[np.nonzero(X[:, best_feature_idx] == feature_thres[0])[0]]
            elif feature_thres[1] == 'right':
                newx = newx[np.nonzero(X[:, best_feature_idx] != feature_thres[0])[0], :]
                newy = newy[np.nonzero(X[:, best_feature_idx] != feature_thres[0])[0]]
            else:
                pass
        return newx, newy, newcolindicator
    
    
    def fit(self, X, y, is_categorical:list=[]):
        # 赋值训练要用到的通用方法
        self._calculate_nodevalue_method = calculate_yproba
        # 赋值信息熵方法
        if self.criterion == "entropy":
            self._calculate_loss = calculate_entropy
            self._search_feature_method = self.__search_feature_entropy
            self._split_dataset_method = self.__split_dataset_entropy
        elif self.criterion == "gini":
            self._calculate_loss = calculate_gini
            self._search_feature_method = self.__search_feature_gini
            self._split_dataset_method = self.__split_dataset_gini
        else:
            pass
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
    # 一、数据准备
    # 数据集1
    X = np.array([[0,0,0,0,0,1,1,1,1,1,2,2,2,2,2],[0,0,1,1,0,0,0,1,0,0,0,0,1,1,0],[0,0,0,1,0,0,0,1,1,1,1,1,0,0,0],[0,1,1,0,0,0,1,1,2,2,2,1,1,2,0]]).T
    y = np.array([0,0,1,1,0,0,0,1,1,1,1,1,1,1,0])
    # 西瓜书数据集2
    columns = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感', '密度', '含糖率', 'label']
    dataSet = [
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '好瓜'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '好瓜'],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, '好瓜'],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, '好瓜'],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, '好瓜'],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, '好瓜'],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, '好瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, '好瓜'],
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, '坏瓜'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, '坏瓜'],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, '坏瓜'],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, '坏瓜'],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, '坏瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, '坏瓜'],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, '坏瓜']
    ]
    #整理出数据集和标签
    X = np.array(dataSet)[:,:8]
    Y = np.array(dataSet)[:,8]
    #对X进行编码
    from sklearn.preprocessing import OrdinalEncoder
    oriencode = OrdinalEncoder(categories='auto')
    oriencode.fit(X[:,:6])
    Xdata=oriencode.transform(X[:,:6])           #编码后的数据
    print(oriencode.categories_)                       #查看分类标签
    Xdata=np.hstack((Xdata,X[:,6:].astype(float)))
    #对Y进行编码
    from sklearn.preprocessing import LabelEncoder
    labelencode = LabelEncoder()
    labelencode.fit(Y)
    Ylabel=labelencode.transform(Y)       #得到切分后的数据
    labelencode.classes_                        #查看分类标签
    labelencode.inverse_transform(Ylabel)    #还原编码前数据
    
    # 二、训练
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

    # CART分类树
    tree_cart_classifier = DecisionTreeClassifier(criterion="gini", minloss=0.0001)
    tree_cart_classifier.fit(Xdata, Ylabel, [True, True, True, True, True, True, False, False])
    tree_cart_classifier.tree
    print_tree(tree_cart_classifier.tree)
    