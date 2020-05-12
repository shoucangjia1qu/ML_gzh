# -*- coding: utf-8 -*-
"""
Created on Sat May  9 12:34:45 2020

@author: ecupl
"""


import numpy as np


class LinearDiscrtAnays(object):
    def __init__(self):
        self.Xi_means = 0               #每个类别的均值向量
        self.Xbar = 0                   #整体的均值向量
        self.covMatrix = []             #每个类别的协方差矩阵
        self.covariance_ = 0            #整体的协方差矩阵
        self.X = 0                      #训练数据
        self.y = 0                      #训练数据的分类标签
        self.classes_ = 0               #具体类别
        self.priors_ = 0                #每个类别的先验概率
        self.n_samples = 0              #训练数据的样本数
        self.n_features = 0             #训练数据的特征数
        self.n_components = 0           #特征数
        self.w = 0                      #特征向量
    
    #初始化特征    
    """calculate params, including:
        0. X, y
        1. n_samples, n_features;
        2. classer, priors_;
        3. Xi_means, Xbar, covMatrix;
    
    """
    def _params_init(self, X, y):
        #0、赋值X和y
        self.X, self.y = X, y
        #1、计算样本数量和特征数量
        self.n_samples, self.n_features = X.shape
        #2、计算类别值、每个类别的先验概率
        self.classes_, yidx = np.unique(y, return_inverse=True)
        self.priors_ = np.bincount(y) / self.n_samples
        #3、计算每类的均值
        means = np.zeros((len(self.classes_), self.n_features))
        np.add.at(means, yidx, X)
        self.Xi_means = means / np.expand_dims(np.bincount(y), 1)
        #4、计算每类的协方差矩阵、整体的协方差矩阵
        self.covMatrix = [np.cov(X[y == group].T) \
                          for idx, group in enumerate(self.classes_)]
        self.covariance_ = sum(self.covMatrix) / len(self.covMatrix)
        #5、计算总体均值向量
        self.Xbar = np.dot(np.expand_dims(self.priors_, axis=0), self.Xi_means)
        return 
    
    #训练
    def train(self, X, y, n_components=None):
        #初始化一些参数
        self._params_init(X, y)
        #求类内平均散度
        Sw = self.covariance_
        #求类间平均散度
        Sb = sum([sum(y == group)*np.dot((self.Xi_means[idx,None] - self.Xbar).T, (self.Xi_means[idx,None] - self.Xbar)) \
                  for idx, group in enumerate(self.classes_)]) / (self.n_samples - 1)
        #SVD求Sw的逆矩阵
        U,S,V = np.linalg.svd(Sw)
        Sn = np.linalg.inv(np.diag(S))
        Swn = np.dot(np.dot(V.T,Sn),U.T)
        SwnSb = np.dot(Swn,Sb)
        #求特征值和特征向量，并取实数部分
        la,vectors = np.linalg.eig(SwnSb)
        la = np.real(la)
        vectors = np.real(vectors)
        #特征值的下标从大到小排列
        laIdx = np.argsort(-la)
        #默认选取(N-1)个特征值的下标
        if n_components == None:
            n_components = len(self.classes_)-1
        #选取特征值和向量
        lambda_index = laIdx[:n_components]
        w = vectors[:,lambda_index]
        self.w = w
        self.n_components = n_components
        return
    
    #求出投影后的矩阵
    def transform(self, X):
        return np.dot(X, self.w)
    
    #预测分类情况，出分类概率
    def predict_prob(self, X):
        #求整体协方差的逆
        Sigma = self.covariance_
        U,S,V = np.linalg.svd(Sigma)
        Sn = np.linalg.inv(np.diag(S))
        Sigman = np.dot(np.dot(V.T,Sn),U.T)
        #线性判别函数值
        value = np.log(np.expand_dims(self.priors_, axis=0)) - \
        0.5*np.multiply(np.dot(self.Xi_means, Sigman).T, self.Xi_means.T).sum(axis=0).reshape(1,-1) + \
        np.dot(np.dot(X, Sigman), self.Xi_means.T)
        return value/np.expand_dims(value.sum(axis=1),1)
    
    #预测分类情况，出具体分类值
    def predict(self, X):
        pValue = self.predict_prob(X)
        return np.argmax(pValue, axis=1)
    
    

#%%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    #0、鸢尾花数据集准备
    from sklearn.datasets import load_iris
    X, y = load_iris(True)
    #1、用自编的LDA算法
    lda_self = LinearDiscrtAnays()
    lda_self.train(X, y)
    #2、直接sklearn实现LDA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    lda_sklearn = LDA(solver="eigen", n_components=2)
    lda_sklearn.fit(X, y)
    #3、降维后的数据集画图
    X_self = lda_self.transform(X)
    X_sklearn = lda_sklearn.transform(X)
    ##自编码的LDA降维后数据
    fig = plt.figure(figsize=(12,4)) 
    ax1 = fig.add_subplot(1,2,1)    
    ax1.scatter(X_self[y==0,0],X_self[y==0,1],c='r',marker='+')
    ax1.scatter(X_self[y==1,0],X_self[y==1,1],c='b',marker='o')
    ax1.scatter(X_self[y==2,0],X_self[y==2,1],c='g',marker='D')
    ax1.set_title("LDA_self")
    ##sklearn的LDA降维后数据
    ax2 = fig.add_subplot(1,2,2)    
    ax2.scatter(X_sklearn[y==0,0],X_sklearn[y==0,1],c='r',marker='+')
    ax2.scatter(X_sklearn[y==1,0],X_sklearn[y==1,1],c='b',marker='o')
    ax2.scatter(X_sklearn[y==2,0],X_sklearn[y==2,1],c='g',marker='D')
    ax2.set_title("LDA_sklearn")
    plt.show()
    #4、再比较下分类情况
    pre_self = lda_self.predict(X)
    pre_sklearn = lda_sklearn.predict(X)
    print("lda_self‘s accurancy is %.3f"%(sum(pre_self==y)/X.shape[0]))
    print("lda_sklearn‘s accurancy is %.3f"%(sum(pre_sklearn==y)/X.shape[0]))

