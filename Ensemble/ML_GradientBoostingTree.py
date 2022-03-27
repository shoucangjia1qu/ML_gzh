# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:26:30 2022

@author: ecupl
"""

import numpy as np
from tools.loss_functions import LeastSquaresError, LeastAbsoluteError
from DecisionTree import DecisionTreeRegression


LOSS_FUNCTIONS = {
    'ls': LeastSquaresError,
    'lad': LeastAbsoluteError,
    # 'huber': HuberLossFunction,
    # 'quantile': QuantileLossFunction,
    # 'deviance': None,  # for both, multinomial and binomial
    # 'exponential': ExponentialLoss,
}


class BaseGradientBoosting():
        
    def __init__(self, loss, learning_rate, n_estimators, 
                 criterion, max_depth, min_samples_leaf, min_criterion_value=0.0001):
        # 模型参数
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        # 单颗树参数
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_criterion_value = min_criterion_value


    def fit(self, X, y):
        # 创建模型
        self.trees_ = []
        # 分配相应的损失函数
        loss_function = LOSS_FUNCTIONS[self.loss]
        self.loss_function_ = loss_function()
        # 初始化预测值为0
        y_prediction = np.zeros(y.shape)
        for i in range(self.n_estimators):
            y_prediction_copy = y_prediction.copy()
            # 逐棵树进行训练
            tree = self._fit_step(X, y, y_prediction_copy)
            self.trees_.append(tree)
            # 根据训练结果更新最新的预测函数
            y_prediction = y_prediction_copy + self.learning_rate*self.trees_[i].predict(X)
            print(f'第{i}棵树的Loss：', self.loss_function_(y, y_prediction))
        return
    
    
    def _fit_step(self, X, y, y_prediction):
        # 1. 计算负梯度
        residual = self.loss_function_.negative_gradient(y, y_prediction)
        # 生成树
        tree = DecisionTreeRegression(self.criterion, self.max_depth, self.min_samples_leaf, self.min_criterion_value)
        # 2. 拟合树
        tree.fit(X, residual)
        # 计算每个样本的叶子结点
        terminal_samples_nodeid = tree.apply(X)
        # 3. 更新更新叶子结点区域的值
        self.loss_function_.update_terminal_regions(tree, terminal_samples_nodeid, X, y, y_prediction)        
        return tree
    
    
    def predict(self, X):
        pass
    
    

# 梯度回归树
class GBRegressionTree(BaseGradientBoosting):
    def __init__(self, loss='ls', learning_rate=0.1, n_estimators=100, 
                 criterion='mse', max_depth=3, min_samples_leaf=1, min_criterion_value=0.0001):
        super(GBRegressionTree, self).__init__(loss, learning_rate, n_estimators, 
                 criterion, max_depth, min_samples_leaf, min_criterion_value)


            

if __name__ == "__main__":
    # 回归树测试
    ## 波士顿房价数据训练
    from sklearn.datasets import load_boston
    X, y = load_boston(True)
    gbrt = GBRegressionTree()
    gbrt.fit(X, y)


