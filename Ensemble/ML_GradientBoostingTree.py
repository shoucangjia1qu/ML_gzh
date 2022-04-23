# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:26:30 2022

@author: ecupl
"""

import numpy as np
from tools.loss_functions import LeastSquaresError, LeastAbsoluteError, HuberLossFunction, BinomialLog
from DecisionTree import DecisionTreeRegression


LOSS_FUNCTIONS = {
    'ls': LeastSquaresError,
    'lad': LeastAbsoluteError,
    'huber': HuberLossFunction,
    'bianry_deviance': BinomialLog,
    # 'multiple_deviance': MultinomialLog,
    # 'exponential': ExponentialLoss,
}


class BaseGradientBoosting():
        
    def __init__(self, loss, learning_rate, n_estimators, 
                 criterion, max_depth, min_samples_leaf, min_criterion_value=0.0001, alpha=0.9):
        # 模型参数
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        # 单颗树参数
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_criterion_value = min_criterion_value
        # Huber参数
        self.alpha = alpha


    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.trees_ = []
        # 分配相应的损失函数
        loss_function = LOSS_FUNCTIONS[self.loss]
        if self.loss in ("huber"):
            self.loss_function_ = loss_function(alpha=self.alpha)
        else:
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
        self.loss_function_.update_terminal_regions(tree.tree, terminal_samples_nodeid, X, y, y_prediction)        
        return tree
    
    
    def predict(self, X):
        y_prediction = np.zeros(X.shape[0])
        for tree in self.trees_:
            y_prediction += self.learning_rate * tree.predict(X)
        return y_prediction
    
    

# 梯度回归树
class GBRegressionTree(BaseGradientBoosting):
    def __init__(self, loss='ls', learning_rate=0.1, n_estimators=100, 
                 criterion='mse', max_depth=3, min_samples_leaf=1, min_criterion_value=0.0001, alpha=0.9):
        super(GBRegressionTree, self).__init__(loss, learning_rate, n_estimators, 
                 criterion, max_depth, min_samples_leaf, min_criterion_value, alpha)



# 梯度分类树
class GBClassficationTree(BaseGradientBoosting):
    def __init__(self, loss='bianry_deviance', learning_rate=0.1, n_estimators=100, 
                 criterion='mse', max_depth=3, min_samples_leaf=1, min_criterion_value=0.0001, alpha=0.9):
        super(GBClassficationTree, self).__init__(loss, learning_rate, n_estimators, 
                 criterion, max_depth, min_samples_leaf, min_criterion_value, alpha)
        
    def predict_proba(self, X):
        y_prediction = super(GBClassficationTree, self).predict(X)
        y_proba = self.loss_function_._raw_predict_proba(y_prediction)
        return y_proba
    
    def predict(self, X):
        y_prediction = super(GBClassficationTree, self).predict(X)
        y_label = self.loss_function_._raw_predict_label(y_prediction)
        return y_label
    
    
    
if __name__ == "__main__":
    # 回归树测试
    ## 波士顿房价数据训练
    from sklearn.datasets import load_boston
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    X, y = load_boston(True)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)
    
    
    # LS
    ## 自编的
    gbrt_ls = GBRegressionTree(loss='ls', n_estimators=100)
    gbrt_ls.fit(train_X, train_y)A
    ypre_ls = gbrt_ls.predict(test_X)
    
    ## sklearn的
    sk_gbrt_ls = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, 
                 criterion='mse', max_depth=3, min_samples_leaf=1)
    sk_gbrt_ls.fit(train_X, train_y)
    ypre_sk_ls = sk_gbrt_ls.predict(test_X)
    
    ## 对比效果
    r2_ls = metrics.r2_score(test_y, ypre_ls)
    r2_sk_ls = metrics.r2_score(test_y, ypre_sk_ls)
    loss_ls = metrics.mean_squared_error(test_y, ypre_ls)
    loss_sk_ls = metrics.mean_squared_error(test_y, ypre_sk_ls)
    print("LS损失函数效果对比：\n", 
          f"R2:自己写的R2={round(r2_ls, 3)}, sklearn的R2={round(r2_sk_ls, 3)}\n", 
          f"MSE:自己写的MSE={round(loss_ls, 3)}, sklearn的MSE={round(loss_sk_ls, 3)}")
    
    
    # LAD
    ## 自编的
    gbrt_lad = GBRegressionTree(loss='lad', n_estimators=100)
    gbrt_lad.fit(train_X, train_y)
    ypre_lad = gbrt_lad.predict(test_X)

    ## sklearn的
    sk_gbrt_lad = GradientBoostingRegressor(loss='lad', learning_rate=0.1, n_estimators=100, 
                 criterion='mse', max_depth=3, min_samples_leaf=1)
    sk_gbrt_lad.fit(train_X, train_y)
    ypre_sk_lad = sk_gbrt_lad.predict(test_X)
    
    ## 对比效果
    r2_lad = metrics.r2_score(test_y, ypre_lad)
    r2_sk_lad = metrics.r2_score(test_y, ypre_sk_lad)
    loss_lad = metrics.mean_squared_error(test_y, ypre_lad)
    loss_sk_lad = metrics.mean_squared_error(test_y, ypre_sk_lad)
    print("LAD损失函数效果对比：\n", 
          f"R2:自己写的R2={round(r2_lad, 3)}, sklearn的R2={round(r2_sk_lad, 3)}\n", 
          f"MSE:自己写的MSE={round(loss_lad, 3)}, sklearn的MSE={round(loss_sk_lad, 3)}")


    # Huber
    gbrt_huber = GBRegressionTree(loss='huber', n_estimators=100, alpha=0.8)
    gbrt_huber.fit(train_X, train_y)
    ypre_huber = gbrt_huber.predict(test_X)

    ## sklearn的
    sk_gbrt_huber = GradientBoostingRegressor(loss='huber', learning_rate=0.1, n_estimators=100, 
                 criterion='mse', max_depth=3, min_samples_leaf=1, alpha=0.8)
    sk_gbrt_huber.fit(train_X, train_y)
    ypre_sk_huber = sk_gbrt_huber.predict(test_X)
    
    ## 对比效果
    r2_huber = metrics.r2_score(test_y, ypre_huber)
    r2_sk_huber = metrics.r2_score(test_y, ypre_sk_huber)
    loss_huber = metrics.mean_squared_error(test_y, ypre_huber)
    loss_sk_huber = metrics.mean_squared_error(test_y, ypre_sk_huber)
    print("Huber损失函数效果对比：\n", 
          f"R2:自己写的R2={round(r2_huber, 3)}, sklearn的R2={round(r2_sk_huber, 3)}\n", 
          f"MSE:自己写的MSE={round(loss_huber, 3)}, sklearn的MSE={round(loss_sk_huber, 3)}")
    


    # 分类树测试
    ## 自带乳腺癌数据
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from scipy.stats import ks_2samp
    X, y = load_breast_cancer(True)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)
    
    # Binomial Deviance
    ## 自编的
    gbct_bianry = GBClassficationTree(loss='bianry_deviance', n_estimators=100)
    gbct_bianry.fit(train_X, train_y)
    yproba_binary = gbct_bianry.predict_proba(test_X)[:, 1]
    ypre_bianry = gbct_bianry.predict(test_X)
    
    ## sklearn的
    sk_gbct_binary = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, 
                 criterion='mse', max_depth=3, min_samples_leaf=1)
    sk_gbct_binary.fit(train_X, train_y)
    yproba_sk_binary = sk_gbct_binary.predict_proba(test_X)[:, 1]
    ypre_sk_binary = sk_gbct_binary.predict(test_X)
    
    ## 对比效果
    auc_binary = metrics.roc_auc_score(test_y, yproba_binary)
    auc_sk_binary = metrics.roc_auc_score(test_y, yproba_sk_binary)
    ks_binary = ks_2samp(yproba_binary[test_y == 1], yproba_binary[test_y != 1]).statistic
    ks_sk_binary = ks_2samp(yproba_sk_binary[test_y == 1], yproba_sk_binary[test_y != 1]).statistic
    acc_binary = metrics.accuracy_score(test_y, ypre_bianry)
    acc_sk_binary = metrics.accuracy_score(test_y, ypre_sk_binary)
    print("Binomial Log-Likelihood损失函数效果对比：\n", 
          f"AUC:自己写的AUC={round(auc_binary, 3)}, sklearn的AUC={round(auc_sk_binary, 3)}\n", 
          f"KS:自己写的KS={round(ks_binary, 3)}, sklearn的KS={round(ks_sk_binary, 3)}\n",
          f"Accuracy:自己写的Accuracy={round(acc_binary, 3)}, sklearn的Accuracy={round(acc_sk_binary, 3)}\n")




    