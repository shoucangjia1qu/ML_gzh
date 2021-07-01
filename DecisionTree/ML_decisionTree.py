# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 22:40:19 2021

@author: ecupl
"""


import numpy as np
from utils import calculate_entropy, calculate_yproba


# from sklearn.tree import DecisionTreeClassifier


# 树结点类
class TreeNode(object):
    """
    保存决策树的结点信息：
    1. 划分特征
    2. 划分特征的值
    3. 结点的值（分类是各类别的概率，回归是具体的值）
    4. 结点的深度
    5. 结点样本数量
    6. 子结点
    """
    def __init__(self, best_feature, thres_list, node_value, node_depth, node_samples, child_node=None):
        self.feature_i = best_feature
        self.feature_thres = thres_list
        self.value = node_value
        self.depth = node_depth
        self.n_samples = node_samples
        self.childNode = child_node



# 树生成类
class BaseTree(object):
    
    def __init__(self, criterion, minloss):
        """
        基础树的类，包括树的构造过程，树的预测过程等等。

        Parameters
        ----------
        criterion : String
            树生成的准则.
        minloss : float
            树停止生长的最小误差，分类树和回归树以及不同的准则会有不同的损失计算函数.

        Returns
        -------
        None
        
        """
        self.criterion = criterion
        self.minloss = minloss
        self.tree = None
        self._calculate_loss = None                    #结点Loss的计算方法
        self._calculate_nodevalue_method = None        #结点值的计算方法
        self._search_feature_method = None             #结点特征选择方法
        self._split_dataset_method = None              #结点样本拆分方法
    
    def _build_tree(self, X, y, node_depth=0, column_name=None):
        """
        树的构造方法。

        Parameters
        ----------
        X : 2D-Array
            该结点上的样本.
        y : 1D-Array
            该结点上的标签，分类结果或者数值.
        node_depth : Int
            该结点的树深.
        column_name : List
            该结点上样本的列名称.

        Returns
        -------
        node : Class
            该结点的信息，无法再分返回None.

        """
        
        node_value = self._calculate_nodevalue_method(y)
        n_samples, n_features = X.shape        
        # 划分树的话，另child_node为None，条件如下：
        ## 1. 无特征可分，
        ## 2. 分类问题中信息熵很低，或者纯度很高，即loss很小的时候；
        ## 3. 回归问题中loss很小的时候
        loss = self._calculate_loss(y)
        if (n_features == 0) or (loss <= self.minloss):
            node = TreeNode(None, None, node_value, node_depth, n_samples, None)
            return node
        
        # 不停止划分的话，另child_node为字典：
        child_node = {}
        ## 1.找到树的最优特征和划分方法
        best_feature, thres_list = self._search_feature_method(X, y, column_name)
        ## 2.遍历保存树的子结点
        for thres in thres_list:
            newx, newy, newcolumnname = self._split_dataset_method(X, y, best_feature, thres, column_name)
            new_depth = node_depth + 1
            child_node[thres] = self._build_tree(newx, newy, new_depth, newcolumnname)
        node = TreeNode(best_feature, thres_list, node_value, node_depth, n_samples, child_node)
        return node


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
    
    
    def __search_feature_entropy(self, X, y, column_name=None):
        # 根据特征值、标签值、根信息熵计算信息增益
        def calculate_gain_entropy(xarray, yarray, root_entropy):
            xSet = np.unique(xarray)
            Di = [sum(xarray==value)/len(xarray) for value in xSet]
            Child_entropy = [calculate_entropy(y[np.nonzero(xarray==value)[0]]) for value in xSet]
            gain_entropy = root_entropy - np.dot(Di, Child_entropy)            
            return gain_entropy, list(xSet)
        # 遍历特征，找到最大信息增益的特征
        n_samples, n_features = X.shape
        best_gain_entropy = 0
        best_feature_index = None
        best_thres_list = None
        entropy0 = calculate_entropy(y)
        for idx in range(n_features):
            gain_entropy, thres_list = calculate_gain_entropy(X[:,idx], y, entropy0)
            if gain_entropy > best_gain_entropy:
                best_gain_entropy = gain_entropy
                best_feature_index = idx
                best_thres_list = thres_list
        return column_name[best_feature_index], best_thres_list
    
    
    def __split_dataset_entropy(self, X, y, best_feature, feature_thres, column_name):
        best_feature_idx = column_name.index(best_feature)
        newx, newy, newcolumnname = np.copy(X), np.copy(y), list(np.copy(column_name))
        newx = np.delete(newx[np.nonzero(X[:,best_feature_idx]==feature_thres)[0],:], best_feature_idx, axis=1)
        newy = newy[np.nonzero(X[:,best_feature_idx]==feature_thres)[0]]
        newcolumnname.remove(best_feature)
        return newx, newy, newcolumnname

    
    def fit(self, X, y):
        # 赋值训练要用到的方法
        self._calculate_loss = calculate_entropy
        self._calculate_nodevalue_method = calculate_yproba
        self._search_feature_method = self.__search_feature_entropy
        self._split_dataset_method = self.__split_dataset_entropy
        # 构建树
        self.tree = self._build_tree(X, y, column_name=list(range(X.shape[1])))
        


def print_tree(tree):
    print(f"========第{tree.depth}层========")
    print(f"该树的结点类型为{tree.value}")
    if tree.childNode is None:
        return
    print(f"最优划分特征是{tree.feature_i}，划分值有{tree.feature_thres}")
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
    
    
    