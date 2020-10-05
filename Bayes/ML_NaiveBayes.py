# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 15:48:21 2020

@author: ecupl
"""

import os
import numpy as np

######一、朴素贝叶斯分类器
class NBayes(object):
    
    #设置属性
    def __init__(self):
        self.trainSet = 0               #训练集数据
        self.trainLabel = 0             #训练集标记
        self.yProba = {}                #先验概率容器
        self.xyProba = {}               #条件概率容器
        self.ySet = {}                  #标记类别对应的数量
        self.ls = 1                     #加入的拉普拉斯平滑的系数
        self.n_samples = 0              #训练集样本数量
        self.n_features = 0             #训练集特征数量


    #计算P(y)先验概率
    def calPy(self, y, LS=True):
        """
        计算先验概率，也就是每个标记的占比

        Parameters
        ----------
        y : 1D array-like
            trainLabel.
        LS : bool, optional
            Weather Laplace Smoothing. The default is True.

        Returns
        -------
        None.

        """
        Py = {}
        yi = {}
        ySet = np.unique(y)
        for i in ySet:
            Py[i] = (sum(y == i) + self.ls) / (self.n_samples + len(ySet))
            yi[i] = sum(y == i)
        self.yProba = Py
        self.ySet = yi
        return


    #计算P(x|y)条件概率
    def calPxy(self, X, y, LS=True):
        """
        计算先验概率，也就是每类分类中，每个变量值的占比

        Parameters
        ----------
        X : 2D array-like
            trainSet.
        y : 1D array-like
            trainLabel.
        LS : bool, optional
            Weather Laplace Smoothing. The default is True.

        Returns
        -------
        None.

        """
        Pxy = {}
        for yi, yiCount in self.ySet.items():
            Pxy[yi] = {}                                                            #第一层是标签Y的分类
            for xIdx in range(self.n_features):
                Pxy[yi][xIdx] = {}                                                  #第二层是不同的特征
                #下标为第xIdx的特征数据
                Xi = X[:, xIdx]
                XiSet = np.unique(Xi)
                XiSetCount = XiSet.size
                #下标为第xIdx，并标签为yi的特征数据
                Xiyi = X[np.nonzero(y==yi)[0], xIdx]
                for xi in XiSet:
                    Pxy[yi][xIdx][xi] = self.classifyProba(xi, Xiyi, XiSetCount)    #第三层是变量Xi的分类概率，离散变量
        self.xyProba = Pxy
        return


    #离散变量直接计算概率
    def classifyProba(self, x, xArr, XiSetCount):
        Pxy = (sum(xArr == x) + self.ls) / (xArr.size + XiSetCount)    #加入拉普拉斯修正的概率
        return Pxy
        
    
    #训练
    def train(self, X, y):
        self.n_samples, self.n_features = X.shape
        #计算先验概率
        self.calPy(y)
        print('P(y)训练完毕!')
        #计算条件概率
        self.calPxy(X, y)
        print('P(x|y)训练完毕!')
        self.trainSet = X
        self.trainLabel = y
        return
      
    
    #预测
    def predict(self, X):
        m, n = X.shape
        proba = np.zeros((m, len(self.yProba)))
        for i in range(m):
            for idx, (yi, Py) in enumerate(self.yProba.items()):
                proba_idx = Py
                for xIdx in range(n):
                    xi = X[i, xIdx]
                    proba_idx *= self.xyProba[yi][xIdx][xi]
                proba[i, idx] = proba_idx
        return proba
    


#%%
if __name__ == "__main__":
    #数据集准备，西瓜书
    from sklearn.preprocessing import OrdinalEncoder
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
    #特征值列表
    labels = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感', '密度', '含糖率']
    dataX = np.array(dataSet)[:,:6]
    oriencode = OrdinalEncoder(categories='auto')
    oriencode.fit(dataX)
    X=oriencode.transform(dataX)
    y = np.array(dataSet)[:,8]
    y[y=="好瓜"]=1
    y[y=="坏瓜"]=0
    y=y.astype(float)
    
    #训练
    NB = NBayes()
    NB.train(X, y)
    Proba = NB.predict(X)
    yPredict = np.argmax(Proba, axis=1)
    #打印先验概率
    print(NB.yProba)
    #打印条件概率
    print(NB.xyProba[0])
    print(NB.xyProba[1])




