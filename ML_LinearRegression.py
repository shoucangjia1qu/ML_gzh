# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class LinearRegression(object):
    """simple linear regression & multivariate linear regression"""
    def __init__(self):
        self.w = 0                  #斜率
        self.b = 0                  #截距
        self.sqrLoss = 0            #最小均方误差
        self.trainSet = 0           #训练集特征
        self.label = 0              #训练集标签
        self.learning_rate = None   #学习率
        self.n_iters = None         #实际迭代次数
        self.lossList = []          #梯度下降每轮迭代的误差列表
    
    def train(self, X, y, method, learning_rate=0.1, n_iters=1000):
        """fit model
        
        Parameters:
        -----------
        X: array-like, 2D shape
            训练集特征
        y: array-like, 1D shape
            训练集标签
        method: formula, matrix, gradient
            训练的方法，可选公式求解法，矩阵求解法，梯度下降法
            公式求解法仅适用于一元线性回归(simple linear regression)
        
        results:
        --------
        w: 1D array or float
            线性回归系数，以array形式返回
            若为一元线性回归，则返回float
        b: float
            线性回归的截距，返回float
        sqrLoss: float
            平方损失和，训练集实际值与预测值的误差平方和
        lossList: List[float]
            记录每轮迭代的平方损失和列表，仅限梯度下降法生成
        """
        if X.ndim < 2:
            raise ValueError("X must be 2D array-like!")
        self.trainSet = X
        self.label = y
        if method.lower() == "formula":
            self.__train_formula()
        elif method.lower() == "matrix":
            self.__train_matrix()
        elif method.lower() == "gradient":
            self.__train_gradient(learning_rate, n_iters)
        else:
            raise ValueError("method value not found!")
        return
            
    #公式求解法(仅适用于一元线性回归)
    def __train_formula(self):
        n_samples, n_features = self.trainSet.shape
        X = self.trainSet.flatten()
        y = self.label
        Xmean = np.mean(X)
        ymean = np.mean(y)
        #求w
        self.w = (np.dot(X, y) - n_samples*Xmean*ymean)/(np.power(X,2).sum() - n_samples*Xmean**2)
        #求b
        self.b = ymean - self.w*Xmean
        #求误差
        self.sqrLoss = np.power((y-np.dot(X,self.w) - self.b), 2).sum()
        return
    
    #矩阵求解法
    def __train_matrix(self):
        n_samples, n_features = self.trainSet.shape
        X = self.trainSet
        y = self.label
        #合并w和b，在X尾部添加一列全是1的特征
        X2 = np.hstack((X, np.ones((n_samples, 1))))
        #求w和b
        EX = np.linalg.inv(np.dot(X2.T,X2))
        what = np.dot(np.dot(EX,X2.T),y)
        self.w = what[:-1]
        self.b = what[-1]
        self.sqrLoss = np.power((y-np.dot(X2,what).flatten()), 2).sum()
        return
        
    #梯度下降法
    def __train_gradient(self, learning_rate, n_iters, minloss=1.0e-6):
        n_samples, n_features = self.trainSet.shape
        X = self.trainSet
        y = self.label
        #初始化迭代次数为0，初始化w0，b0为1，初始化误差平方和以及迭代误差之差
        n = 0
        w = np.ones(n_features)
        b = 1
        sqrLoss0 = np.power((y-np.dot(X,w).flatten()-b), 2).sum()
        self.lossList.append(sqrLoss0)
        deltaLoss = np.inf
        while (n<n_iters) and (sqrLoss0>minloss) and (abs(deltaLoss)>minloss):
            #求w和b的梯度
            ypredict = np.dot(X, w) + b
            gradient_w = -1.*np.dot((y - ypredict), X)/n_samples
            gradient_b = -1.*sum(y - ypredict)/n_samples
            #更新w和b
            w = w - learning_rate * gradient_w
            b = b - learning_rate * gradient_b
            #求更新后的误差和更新前后的误差之差
            sqrLoss1 = np.power((y-np.dot(X,w).flatten()-b), 2).sum()
            deltaLoss = sqrLoss0 - sqrLoss1
            sqrLoss0 = sqrLoss1
            self.lossList.append(sqrLoss0)
            n += 1
        print("第{}次迭代，损失平方和为{}，损失前后差为{}".format(n, sqrLoss0, deltaLoss))
        self.w = w
        self.b = b
        self.sqrLoss = sqrLoss0
        self.learning_rate = learning_rate
        self.n_iters = n+1
        return




def simpleLR(w, b, size=100):
    X = np.expand_dims(np.linspace(-10, 10, size), axis=1)
    y = X.flatten()*w + b + (np.random.random(size)-1)*3
    #公式法求解
    lr1 = LinearRegression()
    lr1.train(X, y, method='formula')
    print("【formula方法】\nw:{}, b:{}, square loss:{}".format(lr1.w, lr1.b, lr1.sqrLoss))
    #矩阵法求解
    lr2 = LinearRegression()
    lr2.train(X, y, method='Matrix')
    print("【matrix方法】\nw:{}, b:{}, square loss:{}".format(lr2.w, lr2.b, lr2.sqrLoss))
    #画图
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(X, y)
    ax.plot(X, X*lr2.w+lr2.b, color='r', linewidth=3)
    plt.show()
    return
    
def multivariateLR():
    from sklearn.datasets import load_boston
    X, y = load_boston(True)
    #将特征X标准化，方便收敛
    X = (X - X.mean(axis=0))/X.std(axis=0)
    #矩阵法求解
    lr1 = LinearRegression()
    lr1.train(X, y, method='Matrix')
    print("【formula方法】\nw:{}, b:{}, square loss:{}".format(lr1.w, lr1.b, lr1.sqrLoss))
    #梯度下降法求解
    lr2 = LinearRegression()
    lr2.train(X, y, method='Gradient', learning_rate=0.1, n_iters=5000)
    print("【matrix方法】\nw:{}, b:{}, square loss:{}".format(lr2.w, lr2.b, lr2.sqrLoss))
    #画梯度下降的误差下降图
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(range(lr2.n_iters), lr2.lossList, linewidth=3)
    ax.set_title("Square Loss")
    plt.show()
    return
    
    
if __name__ == "__main__":
    #1、先用公式法和矩阵法测试下一元线性回归
    simpleLR(1.34, 2.08)
    #2、再用矩阵法和梯度下降法测试下多元线性回归
    multivariateLR()

