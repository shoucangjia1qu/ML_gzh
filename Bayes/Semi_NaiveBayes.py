# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 21:36:19 2021

@author: ecupl
"""

from sklearn import metrics
import numpy as np
import sys
sys.path.extend(['E:\公众号\算法推导\贝叶斯', 'E:\公众号\算法推导\图论'])

from ML_NaiveBayes import NBayes 
from GraphTheory import MST


__all__ = ['Aode', 'Tan']


# AODE算法（Averaged One-Dependent Estimator）
class Aode(NBayes):
    
    def AodeTrain(self, X, y, columnsMark):
        """
        分别训练每个属性作为超父属性下的，联合概率、条件概率等等。
        =====================================================
        1、条件概率格式为：
        =====================================================
        {
        分类类别0:{
                超父属性0:{
                    超父属性值0:{
                        连续特征0:(miu:float, sigma:float),
                        离散特征0:{
                            离散特征属性值0:{
                                数据集个数:int,
                                数据集联合条件概率:float,
                            },
                            离散特征属性值1:{
                            }
                        }
                    },
                    
                    超父属性值1:{
                    }
                },
                
                超父属性1:{
                }
            },
        
        分类类别1:{
            }
        }
        =====================================================
        2、联合概率格式为：
        =====================================================
        {
        分类类别0:{
                超父属性0:{
                    超父属性值0:{
                        数据集个数:int,
                        数据集联合概率:float,
                    },
                    
                    超父属性值1:{
                    }
                },
                
                超父属性1:{
                }
            },
        
        分类类别1:{
            }
        }        
        """
        self.n_samples, self.n_features = X.shape
        # 计算类别的先验联合概率
        Pypa = {}
        # 计算联合概率的的条件概率
        Pxypa = {}
        yset = np.unique(y)
        # 第一层是不同的分类
        for yi in yset:
            Pypa[yi] = {}; Pxypa[yi] = {}
            
            # 第二层是不同的超父属性，如果是连续值则，不能当作超父，离散值当作超父属性 
            for paIdx in range(self.n_features):
                if columnsMark[paIdx] == 1:
                    continue
                Pypa[yi][paIdx] = {}; Pxypa[yi][paIdx] = {}
                paset = np.unique(X[:, paIdx])
                
                # 第三层是不同的超父属性的属性值，分离出来对应的Xarr，和yarr
                for pai in paset:
                    yi_pai_idx = np.nonzero((X[:,paIdx]==pai)&(y==yi))
                    
#                    if paIdx==2 and pai==1:
#                        print(yi, '\n', yi_pai_idx)
                    
                    yarr = y[yi_pai_idx]
                    ## 保存类别的先验联合概率
                    Pypa[yi][paIdx][pai] = self.__calyproba(yarr, self.n_samples, len(yset), len(paset))
                    Pxypa[yi][paIdx][pai] = {}
                    
                    # 第四层是不同的其他特征，若是超父属性则跳过，离散归离散统计，连续归连续统计
                    for xiIdx in range(self.n_features):
                        if xiIdx == paIdx:
                            continue
                        allxiset = np.unique(X[:, xiIdx])
                        Xarr = X[yi_pai_idx, xiIdx].flatten()
                        if columnsMark[xiIdx] == 0:
                            ## 保存离散特征的条件概率
                            Pxypa[yi][paIdx][pai][xiIdx] = self.__categorytrain(Xarr, allxiset)
                        else:
                            ## 保存连续特征的条件概率
                            Pxypa[yi][paIdx][pai][xiIdx] = self.__continuoustrain(Xarr)
                        
#                        if xiIdx == 4 and paIdx==2 and pai==1:
#                            print(Xarr)
                        
        print('P(y,pa)训练完毕!')
        print('P(x|y,pa)训练完毕!')
        self.yProba = Pypa
        self.xyProba = Pxypa
        self.trainSet = X
        self.trainLabel = y
        self.columnsMark = columnsMark        
        return
    
    
    # 计算离散特征的条件概率
    def __categorytrain(self, Xarr, xiset):
        pxypa = {}
        for xivalue in xiset:
            pxypa[xivalue] = {}
            pxypa[xivalue]['count'] = sum(Xarr==xivalue) + self.ls
            pxypa[xivalue]['ratio'] = self.classifyProba(xivalue, Xarr, len(xiset))
        return pxypa
    
    # 计算连续特征的均值和标准差
    def __continuoustrain(self, Xarr):
        pxypa = (Xarr.mean(), Xarr.std())
        return pxypa
        
    # 计算先验联合概率
    def __calyproba(self, yarr, ysum, ysetsum, pasetsum):
        yproba = {}
        yproba['count'] = len(yarr) + self.ls
        yproba['ratio'] = (len(yarr) + self.ls) / (ysum + ysetsum * pasetsum)
        return yproba
    
    
    # 预测
    def aodepredict(self, X, minSet=0):
        n_samples, n_features = X.shape
        proba = np.zeros((n_samples, len(self.yProba)))
        for i in range(n_samples):
            for idx, (yi, Padict) in enumerate(self.yProba.items()):
                sumvalue = 0.
                for paIdx, Pavaluedict in Padict.items():
                    subvalue = 1.
                    pavalue = X[i, paIdx]
                    Statsdict = Pavaluedict[pavalue]
                    if Statsdict['count'] <= minSet:
                        continue
                    Pypa = Statsdict['ratio']
                    subvalue *= Pypa
                    Pxypadict = self.xyProba[yi][paIdx][pavalue]
                    for xiIdx, xiparams in Pxypadict.items():
                        xi = X[i, xiIdx]
                        if isinstance(xiparams, dict):
                            Pxypa = xiparams[xi]['ratio']
                        else:
                            if np.isnan(xiparams[0]) or np.isnan(xiparams[1]):
                                Pxypa = 1.0e-5
                            else:
                                miu = xiparams[0]; sigma = xiparams[1] + 1.0e-5
                                Pxypa = np.exp(-(xi-miu)**2/(2*sigma**2))/(np.power(2*np.pi, 0.5)*sigma) + 1.0e-5
                        subvalue *= Pxypa
                    sumvalue += subvalue
                proba[i, idx] = sumvalue
        return proba
    
    
    # 取对数预测
    def aodepredictLog(self, X, minSet=0):
        n_samples, n_features = X.shape
        proba_log = np.zeros((n_samples, len(self.yProba)))
        for i in range(n_samples):
            for idx, (yi, Padict) in enumerate(self.yProba.items()):
                sumvalue = 0.
                for paIdx, Pavaluedict in Padict.items():
                    subvalue = 0.
                    pavalue = X[i, paIdx]
                    Statsdict = Pavaluedict[pavalue]
                    if Statsdict['count'] <= minSet:
                        continue
                    Pypa = Statsdict['ratio']
                    subvalue += np.log(Pypa)
                    Pxypadict = self.xyProba[yi][paIdx][pavalue]
                    for xiIdx, xiparams in Pxypadict.items():
                        xi = X[i, xiIdx]
                        if isinstance(xiparams, dict):
                            Pxypa = xiparams[xi]['ratio']
                        else:
                            if np.isnan(xiparams[0]) or np.isnan(xiparams[1]):
                                Pxypa = 1.0e-5
                            else:
                                miu = xiparams[0]; sigma = xiparams[1] + 1.0e-5
                                Pxypa = np.exp(-(xi-miu)**2/(2*sigma**2))/(np.power(2*np.pi, 0.5)*sigma) + 1.0e-5
                        subvalue += np.log(Pxypa)
                    sumvalue += subvalue
                proba_log[i, idx] = sumvalue
        return proba_log
    










#============================================================================#
#============================================================================# 
        
# TAN算法,树增强型贝叶斯算法（Tree Augmented Naive Bayes）
class Tan(NBayes):
    
    def __init__(self):
        super(Tan, self).__init__()
        self.CMI_dict = dict()                      #条件互信息字典
        self.f_relationship = dict()                #特征依赖关系{子特征：父特征（依赖属性）}
        
    # 计算分类任务的条件互信息
    def __CMI_classfic(self, xi, xj, y):
        """
        根据分类任务的y值，求特征之间的条件互信息

        Parameters
        ----------
        xi : 1D array-like
            数组1.
        xj : 1D array-like
            数组2.
        y : 1D array-like
            分类标签结果数组.

        Returns
        -------
        cmi : Float
            条件互信息的值.

        """
        yset = np.unique(y)
        y_count = y.size
        cmi = 0
        for yi in yset:
            yi_idx = np.nonzero(y==yi)[0]
            yi_count = yi_idx.size
            yi_proba = yi_count/y_count
            arr0 = xi[yi_idx]
            arr1 = xj[yi_idx]
            mi = self.__calDiscreteMutualInformation(arr0, arr1)
            cmi += mi * yi_proba
        return cmi
            
    # 计算两个离散特征的互信息
    def __calDiscreteMutualInformation(self, arr0, arr1):
        """
        两个都是离散变量的数组求互信息

        Parameters
        ----------
        arr0 : 1D array-like
            数组1.
        arr1 : 1D array-like
            数组2.

        Raises
        ------
        ValueError
            arr0's length should be qeual to arr1's length!.

        Returns
        -------
        mi : float
            两组离散数组的互信息.

        """
        # sklearn中现成的
        #from sklearn import metrics
        #metrics.mutual_info_score(arr0, arr1)
        # 自己写的
        mi = 0
        if len(arr0) != len(arr1):
            raise ValueError("arr0's length should be qeual to arr1's length!")
        else:
            n_samples = len(arr0)
            # 数组一的分布统计
            setX0 = np.unique(arr0)
            bincountX0 = {}
            for xi in setX0:
                bincountX0[xi] = sum(arr0 == xi)
            # 数组二的分布统计
            setX1 = np.unique(arr1)
            bincountX1 = {}
            for xj in setX1:
                bincountX1[xj] = sum(arr1 == xj)
            # 数组一和数组二的联合分布统计
            for i in setX0:
                px0 = bincountX0[i]/n_samples
                for j in setX1:
                    px1 = bincountX1[j]/n_samples
                    px0x1 = sum(np.equal(arr0, i)&np.equal(arr1, j))/n_samples
                    if px0x1 == 0:
                        continue
                    mi += px0x1*np.log(px0x1/(px0*px1))
        return mi
    
    # 计算特征之间的条件互信息，并生成字典，目前只生成离散特征之间的依赖关系
    def get_cmidict(self, X, y, columnsMark):
        n_samples, n_features = X.shape
        CMI_dict = dict()
        featureIdx = np.nonzero(np.array(columnsMark)==0)[0]
        for idx, i in enumerate(featureIdx[:-1]):
            for j in featureIdx[idx+1:]:
                #print(i, j)
                CMI_dict[(i, j)] = self.__CMI_classfic(X[:,i], X[:,j], y)
        return CMI_dict
    
    # 获取特征的依赖关系
    def get_relationship(self, features:list, weight:list):
        # 获取最大次数的顶点
        point_count = dict()
        for p0, p1 in features:
            if p0 not in point_count.keys():
                point_count[p0] = 1
            else:
                point_count[p0] += 1
            if p1 not in point_count.keys():
                point_count[p1] = 1
            else:
                point_count[p1] += 1
        pointcntList = sorted(point_count.items(), key=lambda x: x[1], reverse=True)
        maxcntFeature = pointcntList[0][0]
        # 遍历特征，保存依赖关系
        feature_relationship = dict()
        feature_epoch = [maxcntFeature]
        features_index = []
        while len(features_index) < len(features):
            for idx, feature_pair in enumerate(features):
                if idx in features_index:
                    continue
                if feature_pair[0] in feature_epoch:
                    feature_relationship[feature_pair[1]] = feature_pair[0]
                    feature_epoch.append(feature_pair[1])
                    features_index.append(idx)
                elif feature_pair[1] in feature_epoch:
                    feature_relationship[feature_pair[0]] = feature_pair[1]
                    feature_epoch.append(feature_pair[0])
                    features_index.append(idx)
                else:
                    continue
        return feature_relationship
    
    # TAN算法训练
    def TanTrain(self, X, y, columnsMark):
        # 1 根据最大带权生成树找到每个特征的依赖属性，目前只在离散变量之间做了父属性
        ## 1.1 计算互信息字典
        self.CMI_dict = self.get_cmidict(X, y, columnsMark)
        ## 1.2 生成最大带权树
        points = list(set([i[0] for i in self.CMI_dict.keys()]+[i[1] for i in self.CMI_dict.keys()]))
        self.MaxstClass = MST('max', 'Kruskal')
        self.Maxspanningtree = self.MaxstClass.fit_transform(points, self.CMI_dict)
        ## 1.3自定义顶点，使之有向，构建出依赖关系
        self.f_relationship = self.get_relationship([f for f, w in self.Maxspanningtree], [w for f, w in self.Maxspanningtree])
        
        # 2 训练贝叶斯的联合概率、条件概率
        ## 2.1 初始化变量，样本数量、特征数量、标签类别、
        ##     类别的先验联合概率、联合概率的的条件概率
        self.n_samples, self.n_features = X.shape
        #计算类别的先验概率
        self.calPy(y)
        print('P(y)训练完毕!')
        #yset = np.unique(y)
        #Pypa = {}
        Pxypa = {}
        # 第一层是不同的分类
        for yi, yiCount in self.ySet.items():
            Pxypa[yi] = {}
            
            # 第二层是不同的特征，如果有父属性，就接着加一层父属性，如果没有父属性，则按实际情况来
            for xiIdx in range(self.n_features):
                allXiset = np.unique(X[:, xiIdx])
                # 没有父属性的
                if xiIdx not in self.f_relationship.keys():
                    Xiarr = X[np.nonzero(y==yi)[0], xiIdx].flatten()
                    if columnsMark[xiIdx] == 0:
                        ## 保存离散特征的条件概率
                        Pxypa[yi][xiIdx] = self.__categorytrain(Xiarr, allXiset)
                    else:
                        ## 保存连续特征的条件概率
                        Pxypa[yi][xiIdx] = self.__continuoustrain(Xiarr)
                    continue
                
                # 第三层是有父属性的，值为父属性的各类值
                Pxypa[yi][xiIdx] = {}
                paIdx = self.f_relationship[xiIdx]
                paset = np.unique(X[:, paIdx])
                for pai in paset:
                    xi_pai_idx = np.nonzero((X[:,paIdx]==pai)&(y==yi))[0]
                    Xiarr = X[xi_pai_idx, xiIdx].flatten()
                    if columnsMark[xiIdx] == 0:
                        ## 保存离散特征的条件概率
                        Pxypa[yi][xiIdx][pai] = self.__categorytrain(Xiarr, allXiset)
                    else:
                        ## 保存连续特征的条件概率
                        Pxypa[yi][xiIdx][pai] = self.__continuoustrain(Xiarr)
        print('P(x|y,pa)训练完毕!')
        self.xyProba = Pxypa
        self.trainSet = X
        self.trainLabel = y
        self.columnsMark = columnsMark        
        return

    # 计算离散特征的条件概率
    def __categorytrain(self, Xarr, xiset):
        pxypa = {}
        for xivalue in xiset:
            pxypa[xivalue] = {}
            pxypa[xivalue]['count'] = sum(Xarr==xivalue) + self.ls
            pxypa[xivalue]['ratio'] = self.classifyProba(xivalue, Xarr, len(xiset))
        return pxypa
    
    # 计算连续特征的均值和标准差
    def __continuoustrain(self, Xarr):
        pxypa = (Xarr.mean(), Xarr.std())
        return pxypa
        
    # 预测
    def tanpredict(self, X):
        n_samples, n_features = X.shape
        proba = np.zeros((n_samples, len(self.yProba)))
        for i in range(n_samples):
            for idx, (yi, Xidict) in enumerate(self.xyProba.items()):
                probaValue = 1.
                probaValue *= self.yProba[yi]
                for xiIdx, valuedict in Xidict.items():
                    xi = X[i, xiIdx]
                    ## 值不是字典，说明是连续变量
                    if not isinstance(valuedict, dict):
                        miu = valuedict[0]; sigma = valuedict[1] + 1.0e-5
                        Pxypa = np.exp(-(xi-miu)**2/(2*sigma**2))/(np.power(2*np.pi, 0.5)*sigma) + 1.0e-5
                    ## 第三层不是字典，说明特征没有依赖属性
                    elif not isinstance(list(list(valuedict.values())[0].values())[0], dict):
                        Pxypa = valuedict[xi]['ratio']
                    ## 第三层是字典，说明有依赖属性
                    else:
                        pai = X[i, self.f_relationship[xiIdx]]
                        Pxypa = valuedict[pai][xi]['ratio']
                    probaValue *= Pxypa
                proba[i, idx] = probaValue
        return proba
    
    
    # 取对数预测
    def tanpredictLog(self, X):
        n_samples, n_features = X.shape
        proba_log = np.zeros((n_samples, len(self.yProba)))
        for i in range(n_samples):
            for idx, (yi, Xidict) in enumerate(self.xyProba.items()):
                probaValueLog = 0.
                probaValueLog += np.log(self.yProba[yi])
                for xiIdx, valuedict in Xidict.items():
                    xi = X[i, xiIdx]
                    ## 值不是字典，说明是连续变量
                    if not isinstance(valuedict, dict):
                        miu = valuedict[0]; sigma = valuedict[1] + 1.0e-5
                        Pxypa = np.exp(-(xi-miu)**2/(2*sigma**2))/(np.power(2*np.pi, 0.5)*sigma) + 1.0e-5
                    ## 第三层不是字典，说明特征没有依赖属性
                    elif not isinstance(list(list(valuedict.values())[0].values())[0], dict):
                        Pxypa = valuedict[xi]['ratio']
                    ## 第三层是字典，说明有依赖属性
                    else:
                        pai = X[i, self.f_relationship[xiIdx]]
                        Pxypa = valuedict[pai][xi]['ratio']
                    probaValueLog += np.log(Pxypa)
                proba_log[i, idx] = probaValueLog
        return proba_log
    




