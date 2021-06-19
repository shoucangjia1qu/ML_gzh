# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 21:43:43 2019

@author: ecupl
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
os.chdir(r"D:\mywork\test")

#高斯混合聚类模型类
class GaussianMix(object):
    #1、类的属性
    def __init__(self):
        self.trainSet = 0               #数据集
        self.ClusterLabel = 0           #聚类标签
        self.k = 0                      #聚类个数
        self.LL_ValueList = []          #最大似然函数的值列表
        self.AlphaArr = 0               #高斯混合模型混合系数
        self.MiuArr = 0                 #高斯分布函数的均值参数
        self.SigmaArr = 0               #高斯分布函数的协方差参数
        self.m = 0                      #样本数
        self.d = 0                      #样本维度
        
    #2、初始化函数参数
    def initParas(self, x, k):
        """
        input:
            x:样本集,m*d,其中m为样本数,d为样本维度数
            k:需要聚类的个数
        return:
            初始化AlphaArr, MiuArr, SigmaArr
        """
        self.trainSet = x
        self.k = k
        self.m, self.d = np.shape(x)
        AlphaArr0 = np.ones((1,k))/k
        MiuArr0 = x[np.random.randint(0, self.m, k)]
        #在这里固定好了
        #MiuArr0 = x[[5,6,12]]
        SigmaArr0 = np.array([(np.eye(self.d)*0.1).tolist()]*k)
        return AlphaArr0, MiuArr0, SigmaArr0

    
    #3、计算高斯分布函数
    def Gaussian_multi(self, x, miu, sigma):
        """
        多元高斯分布的密度函数
        input:
            x:样本集,m*d,其中m为样本数,d为样本维度数
            miu:该高斯分布的均值,1*d维
            sigma:该高斯分布的标准差,在此为d*d的协方差矩阵
        return:
            distributionArr:返回样本的概率分布1D数组
        """
        distributionArr = np.exp(-0.5*np.sum(np.multiply(np.dot(x-miu, np.linalg.pinv(sigma)), x-miu), axis=1))/\
        (np.power(2*np.pi, 0.5*self.d)*np.linalg.det(sigma)**0.5)
        return distributionArr

    #4、计算观测值y，高斯分布函数参数条件下，观测来自于第k个高斯分布的概率
    def Gama_Prob(self, x, AlphaArr, MiuArr, SigmaArr):
        """
        隐变量概率分布函数
        input:
            x:样本集,m*d,其中m为样本数,d为样本维度数
            AlphaArr:每个高斯模型出现的先验概率,1*k维,k为聚类个数
            MiuArr:每个高斯模型的均值参数,k*d维
            SigmaArr:每个高斯模型的协方差矩阵参数,k*d*d维
        return:
            GamaProbArr:每个样本出现对应每个高斯模型分布概率的矩阵,m*k维
        """
        GaussProbArr = np.zeros((self.m, self.k))
        for i in range(self.k):
            miu = MiuArr[i]
            sigma = SigmaArr[i]
            GaussProbArr[:,i] = self.Gaussian_multi(x, miu, sigma)
        GamaProbArr = np.copy(np.multiply(GaussProbArr, AlphaArr))
        SumGamaProb = np.sum(GamaProbArr, axis=1).reshape(-1,1)
        return (GamaProbArr/SumGamaProb).round(4), GamaProbArr.round(4)
    
    #5、更新高斯分布函数参数
    def updateParas(self, x, GamaProbArr):
        """
        更新高斯分布函数的参数，包括均值、协方差矩阵、混合系数
        input:
            x:样本集,m*d,其中m为样本数,d为样本维度数
            GamaProbArr:高斯分布函数的后验概率,m*k维
        return:
            newMiuArr:更新后的高斯分布函数的均值,k*d维
            newSigmaArr:更新后的高斯分布的协方差矩阵,k*d*d维
            newAlphaArr:更新后的高斯模型的混合系数,1*k维
        """
        SumGamaProb = np.sum(GamaProbArr, axis=0)
        newMiuArr = np.zeros((self.k,self.d))
        newSigmaArr = np.zeros((self.k,self.d,self.d))
        for i in range(self.k):
            Gama = GamaProbArr[:,i].reshape(-1,1)
            #更新均值
            newMiu = np.sum(np.multiply(Gama, x), axis=0)/SumGamaProb[i]
            newMiuArr[i] = newMiu
            #更新协方差矩阵
            newSigma = np.dot(np.multiply(x-newMiu, Gama).T, x-newMiu)/SumGamaProb[i]
            newSigmaArr[i] = newSigma
        newAlphaArr = SumGamaProb.reshape(1,-1)/self.m
        return newMiuArr, newSigmaArr, newAlphaArr

    #6、求对数似然函数值
    def calLLvalue(self, GaussProbArr):
        LLvalue = sum(np.log(GaussProbArr.sum(axis=1)+1.0e-6))
        return LLvalue
    
    #7、训练：判断是否符合停止条件
    def train(self, x, k, iters):
        """
        循环迭代
        input:
            x:样本集,m*d,其中m为样本数,d为样本维度数
            k:聚类个数
            iters:迭代次数
        return:
            ClusterLabel:最终的聚类结果
        """
        #初始化参数
        AlphaArr0, MiuArr0, SigmaArr0 = self.initParas(x, k)
        LLvalue0 = 0                #初始似然函数值
        LLvalueList = []            #最大似然值列表
        for i in range(iters):
            #计算高斯分布模型的后验概率，也就是已知观测下来自于第k个高斯分布函数的概率
            GamaProbArr, GaussProbArr = self.Gama_Prob(x, AlphaArr0, MiuArr0, SigmaArr0)
            #计算聚类结果
            ClusterLabel = np.argmax(GamaProbArr, axis=1)
            #画分布图(适用于二维)
            if self.d == 2:
                if (i%5 == 0) | (i==iters-1):
                    print("第%d轮迭代"%i)
                    self.drawPics(x, MiuArr0, SigmaArr0, ClusterLabel)
            #计算似然函数，并判断是否继续更新
            LLvalue1 = self.calLLvalue(GaussProbArr)
            print('似然值：',LLvalue1)
            if len(LLvalueList) == 0:
                LLvalue0 = LLvalue1
            else:
                LLdelta = LLvalue1 - LLvalue0
                print('似然值提升：',LLdelta)
                if abs(LLdelta)<1.0e-6:
                    break
                else:
                    LLvalue0 = LLvalue1
            LLvalueList.append(LLvalue1)
            #继续迭代，更新函数参数
            MiuArr1, SigmaArr1, AlphaArr1 = self.updateParas(x, GamaProbArr)
            MiuArr0 = np.copy(MiuArr1)
            SigmaArr0 = np.copy(SigmaArr1)
            AlphaArr0 = np.copy(AlphaArr1)
        self.ClusterLabel = ClusterLabel
        self.LL_ValueList = LLvalueList
        plt.plot(range(len(LLvalueList)), LLvalueList)
        plt.show()
        return

    #8-1、画图
    def drawPics(self, x, MiuArr, SigmaArr, Clusters):
        """
        画图，不同聚类类别的点分布，等高图
        input:
            x:样本集,m*d,其中m为样本数,d为样本维度数
            MiuArr, SigmaArr:高斯分布函数的参数
            Clusters:聚类结果
        out:
            散点图+等高分布图
        """
        plt.figure(figsize=(10,6))
        xgrid, ygrid, zgrid = self.calXYZ(x, MiuArr, SigmaArr)
        #c=plt.contour(xgrid,ygrid,zgrid,6,colors='black')
        plt.contour(xgrid,ygrid,zgrid,6,colors='black')
        plt.contourf(xgrid,ygrid,zgrid,6,cmap=plt.cm.Blues,alpha=0.5)
        #plt.clabel(c,inline=True,fontsize=10)
        for i in range(self.k):
            xi = x[Clusters==i,0]
            yi = x[Clusters==i,1]
            plt.scatter(xi, yi)
            plt.scatter(MiuArr[i,0], MiuArr[i,1], c='r', linewidths=5, marker='D')
        plt.show()
        return

    #8-2、计算网格状的高斯分布，用于画等高线
    def calXYZ(self, x, MiuArr, SigmaArr):
        """
        画等高图需要计算X,Y,Z
        input:
            x:样本集,m*d,其中m为样本数,d为样本维度数
            MiuArr, SigmaArr:高斯分布函数的参数
        return:
            xgrid:x的网格坐标
            ygrid:y的网格坐标
            zgrid:(x,y)网格坐标上高斯分布函数的概率
        """
        x1 = np.copy(x[:,0])
        x1.sort()
        y1 = np.copy(x[:,1])
        y1.sort()
        x2,y2 = np.meshgrid(x1,y1)  # 获得网格坐标矩阵
        Gp = np.zeros((self.m,self.m))
        for i in range(self.m):
            for j in range(self.m):
                xi = x2[i,j]
                yi = y2[i,j]
                data = np.array([xi,yi])
                miuList=[]
                for miu, sigma in zip(MiuArr, SigmaArr):   
                    p = np.exp(-0.5*np.dot(np.dot((data-miu).reshape(1,-1), np.linalg.inv(sigma)), (data-miu).reshape(-1,1)))/\
                    np.power(2*np.pi, 0.5*self.d)*np.linalg.det(sigma)**0.5
                    miuList.append(p)
                Gp[i,j] = max(miuList)
        return x2, y2, Gp

#%%
#开始训练
if __name__ == "__main__":
    ##############西瓜集数据4.0
    data = np.array([[0.697,0.460],[0.774,0.376],[0.634,0.264],[0.608,0.318],[0.556,0.215],[0.403,0.237],[0.481,0.149],
                     [0.437,0.211],[0.666,0.091],[0.243,0.267],[0.245,0.057],[0.343,0.099],[0.639,0.161],[0.657,0.198],
                     [0.360,0.370],[0.593,0.042],[0.719,0.103],[0.359,0.188],[0.339,0.241],[0.282,0.257],[0.748,0.232],
                     [0.714,0.346],[0.483,0.312],[0.478,0.437],[0.525,0.369],[0.751,0.489],[0.532,0.472],[0.473,0.376],
                     [0.725,0.445],[0.446,0.459]])
    GMM = GaussianMix()
    k = 3
    GMM.train(data, k, 50)
    Clusters = GMM.ClusterLabel
    labels = GMM.ClusterLabel
    ##############鸢尾花数据，利用聚类结果和实际标签进行比较
    with open(r"D:\mywork\test\data_UCI\iris.data") as f:
        data = f.readlines()
    trainSet = np.array([row.split(',') for row in data[:-1]])
    label = trainSet[:,-1]
    trainSet = trainSet[:,:-1].astype('float')
    k = 3
    GMM = GaussianMix()
    GMM.train(trainSet, k, 100)
    Clusters = GMM.ClusterLabel




