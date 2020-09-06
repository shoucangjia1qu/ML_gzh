# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 02:28:00 2020

@author: ecupl
"""

import numpy as np
import matplotlib.pyplot as plt


#树结构类
class Tree(object):
    def __init__(self, cutColumn=None, cutValue=None):
        """
        Parameters
        ----------
        cutColumn : Int, optional
            切分超平面的特征列. The default is None.
        cutValue : float, optional
            切分超平面的特征值. The default is None.
            
        """
        self.cutColumn = cutColumn
        self.cutValue = cutValue
        self.nums = 0                       #个数
        self.rootNums = 0                   #在切分超平面上面的实例个数
        self.leftNums = 0                   #在切分超平面左侧的实例个数
        self.rightNums = 0                  #在切分超平面右侧的实例个数
        self._tree_left = None              #左侧树结构
        self._tree_right = None             #右侧树结构
        self.depth = 0                      #树的深度


#kd树实现KNN算法
class KNN(object):
    def __init__(self, K=1):
        self.K_neighbor = K
        self.tree_depth = 0
        self.n_samples = 0
        self.n_features = 0
        self.trainSet = 0
        self.label = 0
        self._tree = 0
        
    def cal_cutColumn(self, n_iter):
        return np.mod(n_iter, self.n_features)
    
    def cal_cutValue(self, Xarray):
        if Xarray.__len__() % 2 == 1:
            #单数序列
            cutValue = np.median(Xarray)
        else:
            #双数序列
            cutValue = Xarray[np.argsort(Xarray)[int(Xarray.__len__()/2)]]
        return cutValue    
    
    #造树
    def build_tree(self, X, n_iter=0):
        nums = X.shape[0]
        #不达切分条件，则不生成树，直接返回None
        if nums < 2*self.K_neighbor:
            return None
        #计算切分的列
        cutColumn = self.cal_cutColumn(n_iter)
        Xarray = X[:,cutColumn]
        #计算切分的值
        cutValue = self.cal_cutValue(Xarray)
        #生成当前的树结构
        tree = Tree(cutColumn, cutValue)
        rootIndex = np.nonzero(Xarray==cutValue)[0]
        leftIndex = np.nonzero(Xarray<cutValue)[0]
        rightIndex = np.nonzero(Xarray>cutValue)[0]
        #保存树的结点数量
        tree.nums = nums
        tree.rootNums = len(rootIndex)
        tree.leftNums = len(leftIndex)
        tree.rightNums = len(rightIndex)
        #保存树深，并加1
        tree.depth = n_iter
        n_iter += 1
        #递归添加左侧树枝结构
        X_left = X[leftIndex,:]
        tree._tree_left = self.build_tree(X_left, n_iter)
        #递归添加右侧树枝结构
        X_right = X[rightIndex,:]
        tree._tree_right = self.build_tree(X_right, n_iter)
        return tree
    
    #训练构造kd树
    def fit_tree(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.trainSet = X
        self.label = y
        self._tree = self.build_tree(X)
        return

    #计算欧氏距离
    def caldist(self, X, xi):
        return np.linalg.norm((X-xi), axis=1)    
    
    #计算一堆数据集距离目标点的距离，并返回K个最近值
    def calKneighbor(self, XIndex, xi):
        trainSet = self.trainSet[XIndex,:]
        knnDict = {}
        distArr = self.caldist(trainSet, xi)
        neighborIndex = XIndex[np.argsort(distArr)[:self.K_neighbor]]
        neighborDist = distArr[np.argsort(distArr)[:self.K_neighbor]]
        for i, j in zip(neighborIndex, neighborDist):
            knnDict[i] = j
        return knnDict
    
    #递归搜索树
    def search_tree(self, trainSetIndex, tree, xi):
        trainSet = self.trainSet[trainSetIndex,:]
        #搜索树找到子结点的过程
        if not (tree._tree_left or tree._tree_right):
            self.neighbor = self.calKneighbor(trainSetIndex, xi)
            print("树深度为{}，切分平面为第{}列特征，初始化搜索树结束！找到{}个近邻点".format(tree.depth, tree.cutColumn, self.K_neighbor))
            return
        else:
            cutColumn = tree.cutColumn
            cutValue = tree.cutValue
            #切分平面左边的实例
            chidlLeftIndex = trainSetIndex[np.nonzero(trainSet[:,cutColumn]<cutValue)[0]]
            #切分平面上的实例
            rootIndex = trainSetIndex[np.nonzero(trainSet[:,cutColumn]==cutValue)[0]]
            #切分平面右边的实例
            chidlRightIndex = trainSetIndex[np.nonzero(trainSet[:,cutColumn]>cutValue)[0]]
            if xi[cutColumn] <= cutValue:
                self.search_tree(chidlLeftIndex, tree._tree_left, xi)
                #回退父结点的过程
                #判断目标点到该切分平面的的距离，计算是否相交
                length = abs(tree.cutValue - xi[cutColumn])
                #不相交的话，则继续回退
                if length >= max(self.neighbor.values()):
                    print("树深度为%d，切分平面为第%d列特征，和父结点的切分平面不相交！"%(tree.depth, tree.cutColumn))
                    return
                #相交的话，先是计算分类平面上实例点的距离，再计算另外半边的实例点的距离
                else:
                    targetIndex = list(rootIndex) + list(chidlRightIndex) + list(self.neighbor.keys())
                    self.neighbor = self.calKneighbor(np.array(targetIndex), xi)
                    print("树深度为%d，切分平面为第%d列特征，检测父结点切分平面和另一侧的样本点是否有更小的！"%(tree.depth, tree.cutColumn))
                    return
            else:
                self.search_tree(chidlRightIndex, tree._tree_right, xi)
                #回退父结点进行判断
                length = abs(tree.cutValue - xi[cutColumn])
                if length >= max(self.neighbor.values()):
                    print("树深度为%d，切分平面为第%d列特征，和父结点的切分平面不相交！"%(tree.depth, tree.cutColumn))
                    return
                else:
                    targetIndex = list(rootIndex) + list(chidlLeftIndex) + list(self.neighbor.keys())
                    self.neighbor = self.calKneighbor(np.array(targetIndex), xi)
                    print("树深度为%d，切分平面为第%d列特征，检测父结点切分平面和另一侧的样本点是否有更小的！"%(tree.depth, tree.cutColumn))
                    return

    #搜索kd树
    def transform_tree(self, Xi):
        self.neighbor = dict()
        self.search_tree(np.arange(self.n_samples), self._tree, Xi)
        return self.neighbor





#%%
if __name__ == "__main__":
    #二维平面数据测试
    X = np.array([[2,5,9,4,8,7],[3,4,6,7,1,2]]).T
    y = np.array([0,0,0,1,1,1])
    knn = KNN(K=2)
    knn.fit_tree(X, y)
    Xi = np.array([6,3])
    knn.transform_tree(Xi)
    
    #鸢尾花数据集测试
    from sklearn.datasets import load_iris
    X, y = load_iris(True)
    #线性计算目标集的最小距离下标
    targetX = np.array([5, 3, 1.2, 0.3])
    minDistIndex = np.argsort(np.linalg.norm((X-targetX), axis=1))
    #K=1时
    knn = KNN(K=1)
    knn.fit_tree(X, y)
    knn.transform_tree(targetX)
    #K=2时
    knn = KNN(K=2)
    knn.fit_tree(X, y)
    knn.transform_tree(targetX)
    #K=3时
    knn = KNN(K=3)
    knn.fit_tree(X, y)
    knn.transform_tree(targetX)
    #K=5时
    knn = KNN(K=5)
    knn.fit_tree(X, y)
    knn.transform_tree(targetX)
    #K=10时
    knn = KNN(K=10)
    knn.fit_tree(X, y)
    knn.transform_tree(targetX)















