# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 23:22:41 2022

@author: ecupl
"""




import numpy as np
from tools.utils import calculate_yproba, calculate_mean, CAL_CRITERION
from DecisionTree.search_feature import SEARCH_FEATURE
from DecisionTree.split_data import SPLIT_DATA
from DecisionTree.analyse_tree import ANALYSE_TREE
from DecisionTree.base import BaseTree

# from sklearn.tree import DecisionTreeClassifier
# 分类树



__all__ = ['DecisionTreeClassication', 'DecisionTreeRegression']

class DecisionTreeClassication(BaseTree):
    def __init__(self, criterion, max_depth, min_samples_leaf, min_criterion_value=0.0001):
        """
        分类树的方法类

        Parameters
        ----------
        criterion : String
            分类树的准则，可选"mse", "mae"，默认"mse".
        max_depth : Int
            最大树深
        min_samples_leaf : Int
            叶子结点最小样本数量
        min_criterion_value : Float
            指标停止划分的最小值.

        Returns
        -------
        None.

        """
        super(DecisionTreeClassication, self).__init__(criterion, max_depth, min_samples_leaf, min_criterion_value)
        self._tree_predict_method = ANALYSE_TREE['classifer']
    
    
    def __search_feature_entropy(self, X, y, column_indicator=None):
        # 遍历特征，找到最大信息增益的特征
        n_samples, n_features = X.shape
        best_entropy_gainratio = 0
        bestsplit_feature_index = None
        bestsplit_value = None
        for idx in range(n_features):
            if (self.is_categorical.__len__() > 0 and self.is_categorical[column_indicator[idx]]==False) or self.is_categorical.__len__() == 0:
                # 连续特征求信息增益率
                entropy_gainratio, feature_value = ent_cutoff_continuous(X[:, idx], y)
            else:
                # 分类特征求信息增益率
                entropy_gainratio, feature_value = ent_cutoff_categorical(X[:, idx], y)
            if entropy_gainratio > best_entropy_gainratio:
                best_entropy_gainratio = entropy_gainratio
                bestsplit_feature_index = idx
                bestsplit_value = feature_value
        return column_indicator[bestsplit_feature_index], bestsplit_value
    
    
    def __split_dataset_entropy(self, X, y, best_feature, feature_thres, column_indicator):
        best_feature_idx = column_indicator.index(best_feature)
        newx, newy, newcolindicator = np.copy(X), np.copy(y), list(np.copy(column_indicator))
        if (self.is_categorical.__len__() > 0 and self.is_categorical[best_feature]==False) or self.is_categorical.__len__() == 0:
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
            newx = np.delete(newx[np.nonzero(X[:,best_feature_idx]==feature_thres)[0],:], best_feature_idx, axis=1)
            newy = newy[np.nonzero(X[:,best_feature_idx]==feature_thres)[0]]
            newcolindicator.remove(best_feature)
        return newx, newy, newcolindicator
    
    
    def fit(self, X, y, is_categorical:list=[]):
        # 计算结点值的函数，用均值来表示
        self._calculate_nodevalue_method = calculate_yproba
        # 计算指标值的函数
        self._calculate_criterion_method = CAL_CRITERION[self.criterion]
        # 搜索最优分裂特征和分裂点的函数
        self._search_feature_method = SEARCH_FEATURE[self.criterion]
        # 拆分数据集的函数
        self._split_dataset_method = SPLIT_DATA[self.criterion]
        # 训练
        super(DecisionTreeClassication, self).fit(X, y, is_categorical)
        return
        



# 回归树
class DecisionTreeRegression(BaseTree):
    def __init__(self, criterion, max_depth, min_samples_leaf, min_criterion_value=0.0001):
        """
        回归树的方法类

        Parameters
        ----------
        criterion : String
            分类树的准则，可选"mse", "mae"，默认"mse".
        max_depth : Int
            最大树深
        min_samples_leaf : Int
            叶子结点最小样本数量
        min_criterion_value : Float
            指标停止划分的最小值.

        Returns
        -------
        None.

        """
        super(DecisionTreeRegression, self).__init__(criterion, max_depth, min_samples_leaf, min_criterion_value)
        self._tree_predict_method = ANALYSE_TREE['regressor']
        
    
    def fit(self, X, y, is_categorical:list=[]):
        # 计算结点值的函数，用均值来表示
        self._calculate_nodevalue_method = calculate_mean
        # 计算指标值的函数
        self._calculate_criterion_method = CAL_CRITERION[self.criterion]
        # 搜索最优分裂特征和分裂点的函数
        self._search_feature_method = SEARCH_FEATURE[self.criterion]
        # 拆分数据集的函数
        self._split_dataset_method = SPLIT_DATA[self.criterion]
        # 训练
        super(DecisionTreeRegression, self).fit(X, y, is_categorical)
        return
    
    
    def predict(self, X):
        res = np.zeros(X.shape[0])
        X_index = np.arange(X.shape[0])
        self._tree_predict_method(X, self.tree, X_index, res, self.is_categorical, 'value')
        return res
    
    
    def apply(self, X):
        res = np.zeros(X.shape[0])
        X_index = np.arange(X.shape[0])
        self._tree_predict_method(X, self.tree, X_index, res, self.is_categorical, 'node_id')
        return res
    




def print_tree(tree):
    # columns = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感', '密度', '含糖率']
    print(f"========第{tree.depth}层========")
    print(f"该树的结点类型为{tree.value}")
    print(f"该树的样本为{tree.n_samples}")
    print(f"划分特征和值{tree.feature_i}, {tree.feature_thres}")
    if tree.childNode is None:
        return
    # print(f"最优划分特征是{columns[tree.feature_i]}，划分值有{tree.feature_thres}")
    for i, node in tree.childNode.items():
        print(f"\n当子结点的属性值为{i}时：")
        print_tree(node)




if __name__ == "__main__":
    # 回归树测试
    ## 波士顿房价数据训练
    from sklearn.datasets import load_boston
    X, y = load_boston(True)
    ## 自编的回归树算法
    rt = DecisionTreeRegression(criterion="mse", max_depth=5, min_samples_leaf=5)
    rt.fit(X, y)
    print_tree(rt.tree)
    y_pred = rt.predict(X)
    node_id = rt.apply(X)
    ## sklearn的回归树算法
    from sklearn.tree import DecisionTreeRegressor
    skrt = DecisionTreeRegressor(criterion="mse", max_depth=5, min_samples_leaf=5)
    skrt.fit(X, y)
    split_features = skrt.tree_.feature
    split_thres = skrt.tree_.threshold
    node_samples = skrt.tree_.n_node_samples
    node_value = skrt.tree_.value
    left_tree_value = node_value[skrt.tree_.children_left]
    right_tree_value = node_value[skrt.tree_.children_right]
    sk_pred = skrt.predict(X)
    sk_id = skrt.apply(X)
    
    
    
    
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
    # C4.5对西瓜集数据分类
    from sklearn.tree import DecisionTreeClassifier

    tree_c45_classifier = DecisionTreeClassifier(criterion="entropy")
    tree_c45_classifier.fit(Xdata, Ylabel, [True, True, True, True, True, True, False, False])
    tree_c45_classifier.tree

    # CART分类树
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(True)
    ct = DecisionTreeClassication(criterion="gini", max_depth=5, min_samples_leaf=5)
    ct.fit(X, y, [])
    print_tree(ct.tree)
    
    sk_gini = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_leaf=5)
    sk_gini.fit(X, y)
    node_samples1 = sk_gini.tree_.n_node_samples
    node_value1 = sk_gini.tree_.value
    
    
  
    
    import pandas as pd
    Xdata2 = pd.DataFrame(Xdata)
    Xdata2.iloc[:,:6] = Xdata2.iloc[:,:6].astype(object)
    
    tree_cart_classifier = DecisionTreeClassication(criterion="gini", max_depth=3, min_samples_leaf=1)
    tree_cart_classifier.fit(Xdata, Ylabel, [True, True, True, True, True, True, False, False])
    print_tree(tree_cart_classifier.tree)  
    
    gini2 = DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_leaf=1)
    gini2.fit(Xdata2, Ylabel)
    node_samples2 = gini2.tree_.n_node_samples
    node_value2 = gini2.tree_.value  
    
    
    
    
    