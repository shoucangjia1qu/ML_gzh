# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 13:23:31 2020

@author: ecupl
"""

from ML_NaiveBayes import NBayes
from Semi_NaiveBayes import *
import numpy as np



class Bayes(NBayes):
    #初始化属性，增加
    def __init__(self, algorithm="Naive", solver="Gaussian"):
        """
        贝叶斯的方法类，可选择朴素贝叶斯、半朴素贝叶斯等
        
        Parameters
        ----------
        algorithm : String, optional
            选择的贝叶斯方法，默认是Naive，可选择AODE, TAN
        solver : String, optional
            选择的概率密度函数，默认是Gaussian，可选择

        
        """
        super(Bayes, self).__init__()
        self.algorithm = algorithm
        self.solver = solver
        
    
    #训练参数
    def train(self, X, y, columnsMark):
        if self.algorithm == "Naive":
            self.NaiveTrain(X, y, columnsMark)
        elif self.algorithm == "Aode":
            self.aode = Aode()
            self.aode.AodeTrain(X, y, columnsMark)
            self.yProba = self.aode.yProba
            self.xyProba = self.aode.xyProba
            self.trainSet = X
            self.trainLabel = y
            self.columnsMark = columnsMark 
        elif self.algorithm == "Tan":
            self.tan = Tan()
            self.tan.TanTrain(X, y, columnsMark)
            self.yProba = self.tan.yProba
            self.xyProba = self.tan.xyProba
            self.trainSet = X
            self.trainLabel = y
            self.columnsMark = columnsMark 

        else:
            pass
    
    
    #预测
    def predict(self, X):
        if self.algorithm == "Naive":
            proba = self.naivepredict(X)
        elif self.algorithm == "Aode":
            proba = self.aode.aodepredict(X, 0)
        elif self.algorithm == "Tan":
            proba = self.tan.tanpredict(X)
        else:
            pass
        return proba
    
    
    #取对数预测
    def predictLog(self, X):
        if self.algorithm == "Naive":
            proba_log = self.naivepredictLog(X)
        elif self.algorithm == "Aode":
            proba_log = self.aode.aodepredictLog(X, 0)
        elif self.algorithm == "Tan":
            proba_log = self.tan.tanpredictLog(X)
        else:
            pass
        return proba_log
        
    
    #朴素贝叶斯训练参数
    def NaiveTrain(self, X, y, columnsMark):
        """
        训练参数的函数，分别训练P(y)的先验概率，离散特征的P(y|X)条件概率，数值型特征的均值、标准差
        

        Parameters
        ----------
        X : 2D array-like
            trainSet.
        y : 1D array-like
            trainLabel.
        columnsMark : list
            各特征的标识，连续变量为1，离散变量为0.

        Returns
        -------
        None.

        """
        self.n_samples, self.n_features = X.shape
        #计算类别的先验概率
        self.calPy(y)
        print('P(y)训练完毕!')
        #区分特征是离散特征还是连续特征，分别计算条件概率或者均值标准差
        Pxy = {}
        for xIdx in range(self.n_features):
            Xarr = X[:, xIdx]
            #第一层是不同的特征
            if columnsMark[xIdx] == 0:
                #特征是离散
                Pxy[xIdx] = self.categoryTrain(Xarr, y)                                                  
            else:
                #特征是连续
                Pxy[xIdx] = self.continuousTrain(Xarr, y)
        print('P(x|y)训练完毕!')
        self.xyProba = Pxy
        self.trainSet = X
        self.trainLabel = y
        self.columnsMark = columnsMark
        return

    
    def categoryTrain(self, Xarr, y, LS=True):
        """
        计算离散特征的条件概率

        Parameters
        ----------
        Xarr : 1D array-like
            trainSet.
        y : 1D array-like
            trainLabel.
        LS : bool, optional
            Weather Laplace Smoothing. The default is True.

        Returns
        -------
        categoryParams : Dict
            分类特征的参数.

        """
        categoryParams = {}
        XiSet = np.unique(Xarr)
        XiSetCount = XiSet.size
        for yi, yiCount in self.ySet.items():
            #第二层是不同的分类标签
            categoryParams[yi] = {}
            Xiyi = Xarr[np.nonzero(y==yi)[0]]
            for xi in XiSet:
                #第三层是变量X的不同值类型
                categoryParams[yi][xi] = self.classifyProba(xi, Xiyi, XiSetCount)    
        return categoryParams
    
    
    def continuousTrain(self, Xarr, y):
        """
        计算数值型特征的均值、标准差等

        Parameters
        ----------
        Xarr : 1D array-like
            trainSet.
        y : 1D array-like
            trainLabel.

        Returns
        -------
        continuousParams : Dict
            数值型特征的参数.

        """
        continuousParams = {}
        for yi, yiCount in self.ySet.items():
            #第二层是不同的分类标签
            Xiyi = Xarr[np.nonzero(y==yi)[0]]
            continuousParams[yi] = (Xiyi.mean(), Xiyi.std())
        return continuousParams
    
    
    #连续变量计算概率密度
    def continuousProba(self, x, miu, sigma):
        if self.solver == "Gaussian":
            Pxy = self.gaussianProba(x, miu, sigma)
        else:
            pass
        return Pxy
    
    
    #高斯概率密度
    def gaussianProba(self, x, miu, sigma):
        """
        高斯密度函数计算概率

        Parameters
        ----------
        x : Float
            当前特征的值.
        miu : Float
            当前特征的均值.
        sigma : Float
            当前特征的标准差.

        Returns
        -------
        Pxy : Float
            概率.

        """
        Pxy = np.exp(-(x-miu)**2/(2*sigma**2))/(np.power(2*np.pi, 0.5)*sigma)
        return Pxy
    
    
    def naivepredict(self, X):
        m, n = X.shape
        proba = np.zeros((m, len(self.yProba)))
        for i in range(m):
            for idx, (yi, Py) in enumerate(self.yProba.items()):
                proba_idx = Py
                for xIdx in range(n):
                    xi = X[i, xIdx]
                    if self.columnsMark[xIdx] == 0:
                        proba_idx *= self.xyProba[xIdx][yi][xi]
                    else:
                        proba_idx *= self.continuousProba(xi, self.xyProba[xIdx][yi][0], self.xyProba[xIdx][yi][1])
                proba[i, idx] = proba_idx
        return proba


    #防止值溢出，预测时取对数
    def naivepredictLog(self, X):
        m, n = X.shape
        log_proba = np.zeros((m, len(self.yProba)))
        for i in range(m):
            for idx, (yi, Py) in enumerate(self.yProba.items()):
                log_proba_idx = 0
                for xIdx in range(n):
                    xi = X[i, xIdx]
                    if self.columnsMark[xIdx] == 0:
                        log_proba_idx += np.log(self.xyProba[xIdx][yi][xi])
                    else:
                        log_proba_idx += np.log(self.continuousProba(xi, self.xyProba[xIdx][yi][0], self.xyProba[xIdx][yi][1]))
                log_proba[i, idx] = log_proba_idx+np.log(Py)
        return log_proba
    
    
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
    X = oriencode.transform(dataX)
    X2 = np.array(dataSet)[:,6:8].astype(float)
    X = np.hstack((X, X2))
    y = np.array(dataSet)[:,8]
    y[y=="好瓜"]=1
    y[y=="坏瓜"]=0
    y=y.astype(float)


    #训练0, Naive算法训练
    Bs = Bayes(algorithm='Naive')
    Bs.train(X, y, [0, 0, 0, 0, 0, 0, 1, 1])
    Proba = Bs.predict(X)
    logProba = Bs.predictLog(X)
    yPredict = np.argmax(logProba, axis=1)
    print(f"错误{sum(yPredict!=y)}个，准确率为：{sum(yPredict==y)/y.size}")

    #训练1, Aode算法训练
    Bs = Bayes(algorithm='Aode')
    Bs.train(X, y, [0, 0, 0, 0, 0, 0, 1, 1])
    Proba = Bs.predict(X)
    logProba = Bs.predictLog(X)
    yPredict = np.argmax(logProba, axis=1)
    print(f"错误{sum(yPredict!=y)}个，准确率为：{sum(yPredict==y)/y.size}")
    
    #训练1, Tan算法训练
    Bs = Bayes(algorithm='Tan')
    Bs.train(X, y, [0, 0, 0, 0, 0, 0, 1, 1])
    Proba = Bs.predict(X)
    logProba = Bs.predictLog(X)
    yPredict = np.argmax(logProba, axis=1)
    print(f"错误{sum(yPredict!=y)}个，准确率为：{sum(yPredict==y)/y.size}")
