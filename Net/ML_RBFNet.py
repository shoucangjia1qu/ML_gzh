# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 19:15:20 2020

@author: ecupl
"""


import numpy as np
import matplotlib.pyplot as plt


class RBFnetwork(object):
    def __init__(self, hidden_nums, r_w, r_c, r_sigma):
        self.h = hidden_nums        #隐含层神经元个数
        self.w = 0                  #线性权值
        self.c = 0                  #神经元中心点
        self.sigma = 0              #高斯核宽度
        self.r = {"w":r_w, 
                  "c":r_c, 
                  "sigma":r_sigma}  #参数迭代的学习率
        self.errList = []           #误差列表
        self.n_iters = 0            #实际迭代次数
        self.tol = 1.0e-5           #最大容忍误差
        self.X = 0                  #训练集特征
        self.y = 0                  #训练集结果
        self.n_samples = 0          #训练集样本数量
        self.n_features = 0         #训练集特征数量

    #计算径向基距离函数
    def guass(self, sigma, X, ci):
        return np.exp(-np.linalg.norm((X-ci), axis=1)**2/(2*sigma**2))
    
    #将原数据高斯转化成新数据
    def change(self, sigma, X, c):
        newX = np.zeros((self.n_samples, len(c)))
        for i in range(len(c)):
            newX[:,i] = self.guass(sigma[i], X, c[i])
        return newX
    
    #初始化参数
    def init(self):
        sigma = np.random.random((self.h, 1))               #(h,1)
        c = np.random.random((self.h, self.n_features))     #(h,n)
        w = np.random.random((self.h+1, 1))                 #(h+1,1)
        return sigma, c, w
    
    #给输出层的输入加一列截距项
    def addIntercept(self, X):
        return np.hstack((X,np.ones((self.n_samples,1))))

    #计算整体误差
    def calSSE(self, prey, y):
        return 0.5*(np.linalg.norm(prey - y))**2
    
    #求L2范数的平方
    def l2(self, X, c):
        m,n = np.shape(X)
        newX = np.zeros((m, len(c)))
        for i in range(len(c)):
            newX[:,i] = np.linalg.norm((X-c[i]), axis=1)**2
        return newX
    
    #训练
    def train(self, X, y, iters, draw=100):
        self.X = X
        self.y = y.reshape(-1,1)
        self.n_samples, self.n_features = X.shape
        sigma, c, w = self.init()                           #初始化参数
        for i in range(iters):
            ##正向计算过程
            hi_output = self.change(sigma,X,c)              #隐含层输出(m,h)，即通过径向基函数的转换
            yi_input = self.addIntercept(hi_output)         #输出层输入(m,h+1)，因为是线性加权，故将偏置加入
            yi_output = np.dot(yi_input, w)                 #输出预测值(m,1)
            error = self.calSSE(yi_output, y)               #计算误差
            if error < self.tol:
                break
            self.errList.append(error)                      #保存误差
            ##误差反向传播过程
            deltaw = np.dot(yi_input.T, (yi_output-y))      #(h+1,m)x(m,1)
            w -= self.r['w']*deltaw/self.n_samples
            deltasigma = np.divide(np.multiply(np.dot(np.multiply(hi_output,self.l2(X,c)).T, \
                        (yi_output-y)), w[:-1]), sigma**3)  #(h,m)x(m,1)
            sigma -= self.r['sigma']*deltasigma/self.n_samples
            deltac1 = np.divide(w[:-1],sigma**2)            #(h,1)
            deltac2 = np.zeros((1,self.n_features))                       #(1,n)
            for j in range(self.n_samples):
                deltac2 += (yi_output-y)[j]*np.dot(hi_output[j], X[j]-c)
            deltac = np.dot(deltac1,deltac2)                #(h,1)x(1,n)
            c -= self.r['c']*deltac/self.n_samples
            # 拟合过程画图
            if (draw!=0) and ((i+1)%draw==0):
                self.draw_process(X, y, yi_output)
                
        self.c = c
        self.w = w
        self.sigma = sigma
        self.n_iters = i
    
    # 画图
    def draw_process(self, X, y, y_prediction):
        plt.scatter(X, y)
        plt.plot(X, y_prediction,c='r')
        plt.show()
        
    
    #预测
    def predict(self, X):
        hi_output = self.change(self.sigma,X,self.c)    #隐含层输出(m,h)，即通过径向基函数的转换
        yi_input = self.addIntercept(hi_output)         #输出层输入(m,h+1)，因为是线性加权，故将偏置加入
        yi_output = np.dot(yi_input, self.w)            #输出预测值(m,1)
        return yi_output
    
    
#%%
if __name__ == "__main__":
    #拟合Hermit多项式
    X = np.linspace(-5, 5 , 500)[:, np.newaxis]
    y = np.multiply(1.1*(1-X+2*X**2),np.exp(-0.5*X**2))
    rbf = RBFnetwork(50, 0.1, 0.2, 0.1)
    rbf.train(X, y, 1000, draw=50)

    