# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 23:07:54 2020

@author: ecupl
"""

import numpy as np
import matplotlib.pyplot as plt


class net(object):
    def __init__(self, neuron, w, atvfuct):
        self.neuron = neuron            #网络的神经元个数
        self.function = atvfuct         #上层网络到这层网络的激活函数
        self.input = 0                  #网络的输入值，上层网络输出值经激活函数后的值
        self.w = w                      #这层网络到下一层的权重
        self.output = 0                 #这层网络的输出值
        self.grdt = 0                   #上一层网络的误差到这层网络输出的梯度项


#%%
#sigmoid激活函数求导
def logitGrdt(fx):
    return np.multiply(fx, (1-fx))

#恒等激活函数求导
def linearGrdt(fx):
    return np.ones(fx.shape)

#softmax激活函数求导
def softmaxGrdt(fx):
    pass


#%%
class BPnet(object):
    """
    
    parameters
    -----------
    hiddenlayers: int 
        隐藏层的层数。   
    neuronofhidden: list-like 
        隐藏层的神经元个数。
    optfunction: str 
        输出层的激活函数，解决不同问题。
    """
    #1、初始化属性
    def __init__(self, hiddenlayers, hiddenNeurons, optfunction):
        self.hiddenslayers = hiddenlayers           #隐藏层层数
        self.hiddenNeurons = hiddenNeurons         #隐藏层每层的神经元个数
        self.Fuct_output = optfunction              #输出层的激活函数选择，解决不同的问题
        #-------------------------------------------------------------
        self.W_xinput = 0                           #输入层的权重
        self.hiddens = []                           #隐藏层的容器
        self.r = 0                                  #学习率
        self.errList = []                           #误差列表
        self.tol = 1.0e-4                           #可容忍的最小误差
        self.n_iters = 0                            #实际迭代次数
        #-------------------------------------------------------------
        self.trainSet = 0                           #数据集X
        self.label = 0                              #数据标签Y
        self.n_samples = 0                          #样本数量
        self.n_features = 0                         #属性数量
    
    #2、通过激活函数求值    
    def activeFuct(self, z, fuctType):
        if fuctType=="sigmoid":
            return 1.0/(1+np.exp(-z))
        elif fuctType=="linear":
            return z
        elif fuctType=="softmax":
            pass
        else:
            pass

    #3、激活函数求导    
    def grdt_activeFuct(self, fx, fuctType):
        if fuctType=="sigmoid":
            return np.multiply(fx, (1-fx))
        elif fuctType=="linear":
            return np.ones(fx.shape)
        elif fuctType=="softmax":
            pass
        else:
            pass

    #4、定义累计误差函数
    def errorfunc(self, y, ypre):
        return sum(0.5*np.power((ypre-y),2))
    
    #5、初始化神经元权重，(-1, 1)
    def initW(self, row, column):
        return (np.random.random((row+1, column))-1)*2
    
    #6、正向计算输出值
    def calResult(self, W_input, hiddens, X):
        """

        Parameters
        ----------
        W_input : array-like
            输入层的权值.
        hiddens : list, [Class1, Class2, ..., ClassN]
            放置隐藏层的容器.
        X : array-like
            数据集
        
        Returns
        -------
        y_output : arr-like
            正向计算的结果

        """
        y_output = 0
        bias = np.ones((self.n_samples, 1))
        for i in range(self.hiddenslayers+1):
            #print("第%d轮："%i)
            #print("===================")
            if i==0:
                #输入层计算：加入偏置项 -> 计算点乘项(输出项) -> 经过第一层隐藏层激活函数输出
                Xi = np.hstack((X, bias))
                #print("输入的格式",Xi.shape)
                Xi_dot = np.dot(Xi, W_input)
                #print("经过激活函数后的格式",Xi_dot.shape)
                hiddens[i].input = self.activeFuct(Xi_dot, hiddens[i].function)
            elif i==self.hiddenslayers:
                #输出层计算：加入偏置项 -> 计算点乘项(输出项) -> 经过输出层激活函数输出
                Yi = np.hstack((hiddens[i-1].input, bias))
                #print("输入的格式",Yi.shape)
                Yi_dot = np.dot(Yi, hiddens[i-1].w)
                hiddens[i-1].output = Yi_dot
                y_output = self.activeFuct(Yi_dot, self.Fuct_output)
                #print("经过激活函数后的格式",y_output.shape)
            else:
                #隐藏层计算：加入偏置项 -> 计算本层网络的输出项，并保存 -> 经过下层网络的激活函数输出
                Hi = np.hstack((hiddens[i-1].input, bias))
                #print("输入的格式",Hi.shape)
                Hi_dot = np.dot(Hi, hiddens[i-1].w)
                hiddens[i-1].output = Hi_dot
                hiddens[i].input = self.activeFuct(Hi_dot, hiddens[i].function)
                #print("经过激活函数后的格式",hiddens[i].input.shape)
        return y_output
    
    #7、误差反向传播进行优化
    def bwPropagation(self, W_input, hiddens, y, ypre, r):
        bias = np.ones((self.n_samples, 1))
        for i in range(self.hiddenslayers+1)[::-1]:
            if i==self.hiddenslayers:
                #最后一层隐藏层到输出层的权值更新：
                #计算误差 -> 计算输出层的梯度项 -> 更新权值
                error = ypre - y
                grdt_output = np.multiply(error, self.grdt_activeFuct(ypre, self.Fuct_output))
                hiddens[i-1].grdt = grdt_output
                Hi = np.hstack((hiddens[i-1].input, bias))
                w0 = hiddens[i-1].w.copy()
                hiddens[i-1].w = hiddens[i-1].w - r*np.dot(Hi.T, grdt_output)
            elif i==0:
                #输入层到第一层隐藏层的权值更新：
                ##计算误差 -> 计算上一层网络到这层网络的梯度项 -> 更新权值
                grdt_hidden = np.multiply(np.dot(hiddens[i].grdt, w0[:-1,:].T), \
                                          self.grdt_activeFuct(hiddens[i].input, hiddens[i].function))
                Xi = np.hstack((self.trainSet, bias))
                W_input = W_input - r*np.dot(Xi.T, grdt_hidden) 
            else:
                #这层隐藏层到下一层隐藏层的权值更新：
                #计算误差 -> 计算上一层网络到这层网络的梯度项 -> 更新权值
                grdt_hidden = np.multiply(np.dot(hiddens[i].grdt, w0[:-1,:].T), \
                                          self.grdt_activeFuct(hiddens[i].input, hiddens[i].function))
                hiddens[i-1].grdt = grdt_hidden
                Hi = np.hstack((hiddens[i-1].input, bias))
                w0 = hiddens[i-1].w.copy()
                hiddens[i-1].w = hiddens[i-1].w - r*np.dot(Hi.T, grdt_hidden)  
        return W_input
    
            
    #8、训练BP网络
    def train(self, X, y, r, Iters):
        self.n_samples, self.n_features =np.shape(X)
        self.trainSet = X
        self.label = y.reshape(-1,1)
        #6.1 初始化网络权值
        for i in range(self.hiddenslayers+1):
            if i==0:
                #输入层
                W_xinput = self.initW(self.n_features, self.hiddenNeurons[i])
            elif i==self.hiddenslayers:
                #隐藏层
                w_hidden = self.initW(self.hiddenNeurons[i-1], self.label.shape[1])
                hi_layer = net(self.hiddenNeurons[i-1], w_hidden, "sigmoid")
                self.hiddens.append(hi_layer)
            else:
                #隐藏层
                w_hidden = self.initW(self.hiddenNeurons[i-1], self.hiddenNeurons[i])
                hi_layer = net(self.hiddenNeurons[i-1], w_hidden, "sigmoid")
                self.hiddens.append(hi_layer)
        #------------------------------------------------------------------------
        for i in range(Iters):
            #6.2 正向计算输出值
            yi_output = self.calResult(W_xinput, self.hiddens, self.trainSet)
            #6.3 计算整体误差并保存
            SSE = self.errorfunc(self.label, yi_output)
            if SSE<self.tol:
                break
            self.errList.append(SSE[0])
            #6.4 反向传播误差计算梯度并更新权重
            W_input = self.bwPropagation(W_xinput, self.hiddens, self.label, yi_output, r)
        #------------------------------------------------------------------------
        self.r = r
        self.W_input = W_input
        self.n_iters = i

    #预测
    def predict(self, dataSet):
        y_output = 0
        bias = np.ones((dataSet.shape[0], 1))
        for i in range(self.hiddenslayers+1):
            if i==0:
                Xi = np.hstack((dataSet, bias))
                Xi_dot = np.dot(Xi, self.W_input)
                output = self.activeFuct(Xi_dot, self.hiddens[i].function)
            elif i==self.hiddenslayers:
                Yi = np.hstack((output, bias))
                Yi_dot = np.dot(Yi, self.hiddens[i-1].w)
                y_output = self.activeFuct(Yi_dot, self.Fuct_output)
            else:
                Hi = np.hstack((output, bias))
                Hi_dot = np.dot(Hi, self.hiddens[i-1].w)
                output = self.activeFuct(Hi_dot, self.hiddens[i].function)
        return y_output

#%%
if __name__ == "__main__":
    #1 数据准备
    with open(r"D:\mywork\test\book_ML\dataSet_BP.txt") as f:
        data = f.readlines()
    X = np.array([row.split() for row in data]).astype(float)            
    y = X[:,-1].reshape(-1,1)            
    X = X[:,0:-1]
    X = (X-X.mean(axis=0))/X.std(axis=0)      
    #2 实例化BP类
    bp = BPnet(3, [20, 10, 5], "sigmoid")
    bp.train(X, y, 0.1, 2000)
    #3 画等高图看下分类效果
    x = np.linspace(-3,3,50)
    xx = np.ones((50,50))
    xx[:,0:50] = x
    yy = xx.T
    dataSet = np.array([xx.reshape(-1),  yy.reshape(-1)]).T
    zz = bp.predict(dataSet).reshape(50,50)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(X[y[:,0]==0,0], X[y[:,0]==0,1], c='b', marker='D')
    ax.scatter(X[y[:,0]==1,0], X[y[:,0]==1,1], c='r', marker='o')
    ax.contour(x,x,zz,1,colors = 'black')
    plt.show()
    
    
    
    
    
