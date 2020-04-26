# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 10:53:47 2020

@author: ecupl
"""
#修改到ML_LogisticRegression.py文件相对路径
import os
os.chdir(r"E:\公众号\算法推导\逻辑斯蒂回归")
#只导入逻辑回归的算法类
from ML_LogisticRegression import LogisticRegressionSelf as logit
import numpy as np

class allLogitMethod(logit):
    def __init__(self, method):
        """init class
        
        Parameters:
        -----------
        method: optimization method for ML
            优化方法，可选择的有"gradient", "newton", "DFP", "BFGS"
        """
        super().__init__()
        self.method = method
        self.graList = []               #记录梯度模长的列表
    
    #重写下train方法
    def train(self, X, y, n_iters=1000, learning_rate=0.01):
        """fit model
        
        Parameters:
        -----------
        X: array-like, 2D shape
            训练集特征
        y: array-like, 1D shape
            训练集标签
        n_iters: Int, default=1000
            优化方法的迭代次数
        learning_rate: float, default=0.01
            学习率，在梯度下降中用得到
        
        results:
        --------
        w: 1D array
            特征的系数
        b: float
            截距
        accurancy: float
            准确率，根据模型优化后自动生成的准确率
        llList: List[float]
            记录每轮迭代的似然值列表
        """
        if X.ndim < 2:
            raise ValueError("X must be 2D array-like!")
        self.trainSet = X
        self.label = y
        if self.method.lower() == "gradient":
            self._LogisticRegressionSelf__train_gradient(n_iters, learning_rate)
        elif self.method.lower() == "newton":
            self._LogisticRegressionSelf__train_newton(n_iters)
        elif self.method.lower() == "dfp":
            self.__train_dfp(n_iters, learning_rate)
        else:
            raise ValueError("method value not found!")
        return  
    
    #一维搜索法求出最优lambdak，更新W后，使得似然值最小
    def __updateW(self, X, Y, lambdak, W0, Pk):
        """
        此处对lambdak的处理仅简单用1~i次方来逐步变小，以选取到最小似然值的lambdak
        """
        min_LLvalue = np.inf
        W1 = np.zeros(W0.shape)
        for i in range(10):
            Wi = W0 - (lambdak**i)*Pk
            Ypreprob, LLvalue = self.PVandLLV(X, Y, Wi)
            if LLvalue < min_LLvalue:
                min_LLvalue = LLvalue
                W1 = np.copy(Wi)
                deltaW = - (lambdak**i)*Pk
                bestYpreprob = Ypreprob
        return W1, deltaW, min_LLvalue, bestYpreprob
        
    #新增拟牛顿法-DFP优化算法
    def __train_dfp(self, n_iters, learning_rate):
        """Quasi-Newton Method DFP（Davidon-Fletcher-Powell）
        Iteration Process:
        1. 初始化参数W0，初始化海塞矩阵逆矩阵的替代矩阵Dk，计算初始梯度Gk0，计算初始似然值;
        2. 更新参数W1，需要用到一维搜索方法，目标是似然值最小;
        3. 计算梯度Gk1，判断是否结束迭代;
        4. 更新Dk+1，需用到Dk, (W1-W0), (Gk1-Gk0);
        5. 到第2步继续迭代;
        """
        n_samples, n_features = self.trainSet.shape
        X = self.trainSet
        y = self.label
        #合并w和b，在X尾部添加一列全是1的特征
        X2 = np.hstack((X, np.ones((n_samples, 1))))
        #将y转置变为(n_samples,1)的矩阵
        Y = np.expand_dims(y, axis=1)
        #初始化特征系数W，初始化替代对称矩阵
        W = np.zeros((1, n_features+1))
        Dk0 = np.eye(n_features+1)
        #计算初始的预测值、似然值，并记录似然值
        Ypreprob, LL0 = self.PVandLLV(X2, Y, W)
        self.llList.append(LL0)
        #根据初始的预测值计算初始梯度，并记录梯度的模长
        Gk0 = self._LogisticRegressionSelf__calGradient(X2, Y, Ypreprob)
        graLength = np.linalg.norm(Gk0)
        self.graList.append(graLength)
        #初始化迭代次数
        k = 0
        while (k<n_iters) and (graLength>self.tol):
            #计算优化方向的值Pk=Gk0.Dk0
            Pk = np.dot(Gk0, Dk0)
            #一维搜索更新参数，并保存求得的最小似然值
            W, deltaW, min_LLvalue, Ypreprob = self.__updateW(X2, Y, learning_rate, W, Pk)
            self.llList.append(min_LLvalue)
            #更新梯度Gk和deltaG，同时求得梯度的模长和更新前后的模长差值
            Gk1 = self._LogisticRegressionSelf__calGradient(X2, Y, Ypreprob)
            graLength = np.linalg.norm(Gk1)
            self.graList.append(graLength)
            deltaG = Gk1 - Gk0
            Gk0 = Gk1
            #更新替代矩阵Dk
            Dk1 = Dk0 + np.dot(deltaW.T, deltaW)/np.dot(deltaW, deltaG.T) - \
            np.dot(np.dot(np.dot(Dk0, deltaG.T), deltaG), Dk0)/np.dot(np.dot(deltaG, Dk0), deltaG.T)
            Dk0 = Dk1
            k += 1
        self.n_iters = k
        self.w = W.flatten()[:-1]
        self.b = W.flatten()[-1]
        Ypre = np.argmax(np.column_stack((1-Ypreprob,Ypreprob)), axis=1)
        self.accurancy = sum(Ypre==y)/n_samples
        print("第{}次停止迭代，梯度模长为{}，似然值为{}，准确率为{}".format(self.n_iters, self.graList[-1], self.llList[-1], self.accurancy))
        print("w:{};\nb:{}".format(self.w, self.b))
        return

#%%            
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    import time
    X, y = make_classification(n_samples=1000, n_features=4)
    #1、自编的梯度下降法进行拟合
    #logit_gd = allLogitMethod("gradient")
    #logit_gd.train(X, y, n_iters=20000, learning_rate=0.3)
    #plt.plot(range(logit_gd.n_iters+1), logit_gd.llList)
    #plt.show()
    #2、自编的牛顿法进行拟合
    time_nt = time.time()
    logit_nt = allLogitMethod("newton")
    logit_nt.train(X, y, n_iters=100)
    print("迭代时长：", time.time()-time_nt)
    plt.plot(range(logit_nt.n_iters+1), logit_nt.llList)
    plt.show()
    #3、自编的拟牛顿法-DFP算法进行拟合
    time_dfp = time.time()
    logit_dfp = allLogitMethod("DFP")
    logit_dfp.train(X, y, n_iters=20, learning_rate=0.5)
    print("迭代时长：", time.time()-time_dfp)
    fig = plt.figure(figsize=(12,4)) 
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(range(logit_dfp.n_iters+1), logit_dfp.llList)
    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(range(logit_dfp.n_iters+1), logit_dfp.graList)
    plt.show()
    #4、sklearn封装的逻辑回归进行拟合
    #from sklearn.linear_model import LogisticRegression
    #logit_sklearn = LogisticRegression(solver="saga")
    #logit_sklearn.fit(X, y)
    #print("w:{};\nb:{}".format(logit_sklearn.coef_, logit_sklearn.intercept_))
    #print("score:",logit_sklearn.score(X, y))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    