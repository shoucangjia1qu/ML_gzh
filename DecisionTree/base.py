# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 00:04:07 2022

@author: ecupl
"""


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

    def __init__(self, node_id, best_feature, thres_list, node_value, node_depth, node_samples, child_node=None):
        self.node_id = node_id
        self.feature_i = best_feature
        self.feature_thres = thres_list
        self.value = node_value
        self.depth = node_depth
        self.n_samples = node_samples
        self.childNode = child_node


# 树生成类
class BaseTree(object):

    def __init__(self, criterion, max_depth, min_samples_leaf, min_criterion_value=0.0001):
        """
        基础树的类，包括树的构造过程，树的预测过程等等。

        Parameters
        ----------
        criterion : String
            树生成的准则.
        max_depth : Int
            树最大的划分层数.
        min_samples_leaf : Int
            树叶子结点最小样本数量.
        min_criterion_value : Float
            指标停止划分的最小值.

        Returns
        -------
        None

        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_criterion_value = min_criterion_value
        self.node_nums = 0
        self.tree = None
        self._calculate_nodevalue_method = None  # 结点值的计算方法
        self._calculate_criterion_method = None # 结点指标值的计算方法
        self._search_feature_method = None  # 结点特征选择方法
        self._split_dataset_method = None  # 结点样本拆分方法
        self._tree_predict_method = None # 树预测的方法
        
        
    def __build_tree(self, X, y, node_depth=0, column_indicator=None):
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
        column_indicator : List
            该结点上样本的列的指针.

        Returns
        -------
        node : Class
            该结点的信息，无法再分返回None.

        """
        node_value = self._calculate_nodevalue_method(y)
        n_samples, n_features = X.shape
        criterion_value = self._calculate_criterion_method(y)
        # print(X.shape)
        # 划分树的话，另child_node为None，条件如下：
        ## 1. 无特征可分；
        ## 2. 树深小于等于最大值；
        ## 3. 叶子结点数量大于等于最小值;
        ## 4. 计算的指标值低于最小值，比如信息熵、GINI、MSE低于最小值。
        if (n_features == 0) or (node_depth>=self.max_depth) or (criterion_value<=self.min_criterion_value):
            node = TreeNode(self.node_nums, None, None, node_value, node_depth, n_samples, None)
            self.node_nums += 1
            return node
        # 不停止划分的话，令child_node为字典：
        child_node = {}
        ## 1.找到树的最优特征和划分方法，如果找不到划分方法，说明叶子节点不满足最小样本数量的要求，就直接返回
        best_feature, thres_list = self._search_feature_method(X, y, self.min_samples_leaf, self.is_categorical, column_indicator)
        if best_feature is None:
            node = TreeNode(self.node_nums, None, None, node_value, node_depth, n_samples, None)
            self.node_nums += 1
            return node
        
        ## 2.遍历保存树的子结点
        for thres in thres_list:
            newx, newy, newcolindicator = self._split_dataset_method(X, y, best_feature, thres, self.is_categorical, column_indicator)
            # print(node_value, newx.shape)
            new_depth = node_depth + 1
            child_node[thres] = self.__build_tree(newx, newy, new_depth, newcolindicator)
        node = TreeNode(self.node_nums, best_feature, thres_list, node_value, node_depth, n_samples, child_node)
        self.node_nums += 1
        return node
    
    
    def fit(self, X, y, is_categorical:list=[]):
        """
        模型训练

        Parameters
        ----------
        X : 2D-Array
            样本X.
        y : 1D-Array
            样本Label.
        is_categorical : list, optional
            特征是否分类变量. The default is [].

        Returns
        -------
        None.

        """
        self.is_categorical = is_categorical
        # 构建树的参数
        self.tree = self.__build_tree(X, y, column_indicator=list(range(X.shape[1])))
        return

    
    def predict(self, X):
        pass
    
    
    def apply(self, X):
        pass
    
    

    
    
    
    
    