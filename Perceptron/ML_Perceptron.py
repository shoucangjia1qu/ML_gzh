# -*- coding: utf-8 -*-
"""
Created on Sun May 31 19:24:56 2020

@author: ecupl
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



class perceptron(object):
    def __init__(self):
        self.n_samples = 0          #样本数量
        self.n_features = 0         #样本特征
        self.w = 0                  #参数w
        self.b = 0                  #参数b
        self.X = 0                  #训练集X
        self.y = 0                  #训练集分类标签y
        self.epoch = 0              #实际迭代次数
        self.errList = []           #错误率
        self.ypre = 0               #分类Y的预测值
    
    #求函数值
    def calY(self, X, w, b):
        return np.dot(X, w) + b
    
    #训练
    def train(self, X, y, learning_rate, epochs=200):
        #初始化参数
        self.n_samples, self.n_features = X.shape
        w = np.ones((self.n_features, 1))
        b = 1
        for i in range(epochs):
            error = 0
            for n in range(self.n_samples):
                Xi = np.expand_dims(X[n],1)
                yi = y[n]
                yValue = self.calY(Xi.flatten(), w.flatten(), b)
                #对每个样本进行判断是否分类正确
                if yi*yValue <= 0:
                    w = w + learning_rate*Xi*yi
                    b = b + learning_rate*yi
                    error += 1
                else:
                    pass
            self.errList.append(error)
            #无错分就退出循环
            if error == 0:
                break
        self.w = w
        self.b = b
        self.X = X
        self.y = y
        self.epoch = i
        self.ypre = np.sign(self.calY(self.X, self.w, self.b))
        return
        
        

#%%
if __name__ == "__main__":
    #准备数据，3维
    X = np.random.random((500,3))
    y = np.ones((500,1))
    y[(0.55*X[:,0]+0.3*X[:,1]-0.15)>X[:,2]] = -1
    #画原图散点图
    fig = plt.figure()
    ax = Axes3D(fig)
    x1 = X[:,0]; y1 = X[:,1]; z1 = X[:,2]
    ax.scatter(x1[y.flatten()==-1], y1[y.flatten()==-1] ,z1[y.flatten()==-1], color='red')
    ax.scatter(x1[y.flatten()==1], y1[y.flatten()==1] ,z1[y.flatten()==1], color='blue')
    plt.show()
    #训练
    per = perceptron()
    per.train(X, y, 0.005)
    print("参数是：", per.w, per.b)
    print("迭代次数：", per.epoch)
    #画图1：错误个数
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(range(per.epoch+1), per.errList)
    plt.show()
    #画图2：区分平面
    w = per.w; b = per.b
    ##画散点图
    fig = plt.figure()
    ax = Axes3D(fig)
    x1 = X[:,0]; y1 = X[:,1]; z1 = X[:,2]
    ax.scatter(x1[y.flatten()==-1], y1[y.flatten()==-1] ,z1[y.flatten()==-1], color='red')
    ax.scatter(x1[y.flatten()==1], y1[y.flatten()==1] ,z1[y.flatten()==1], color='blue')
    ##画平面图
    x2, y2 = np.meshgrid(x1, y1)
    z2 = -x2*(w[0]/w[2]) - y2*(w[1]/w[2]) -(b/w[2])
    ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap='rainbow', alpha=0.3) 
    x3 = np.linspace(-0.2,1.2,100)
    y3 = np.linspace(-0.2,1.2,100)
    z3 = np.linspace(-0.2,1.2,100)
    x3, y3 = np.meshgrid(x3, y3)
    z3 = -x3*(w[0]/w[2]) - y3*(w[1]/w[2]) -(b/w[2])    
    ax.plot_surface(x3, y3, z3, rstride=1, cstride=1, cmap='BrBG', alpha=0.3) 
    plt.show()
    
    
    
    
    