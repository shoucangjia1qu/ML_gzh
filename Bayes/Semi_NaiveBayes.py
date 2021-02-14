# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 21:36:19 2021

@author: ecupl
"""

from ML_NaiveBayes import NBayes 
import numpy as np

__all__ = ['Aode']


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
    


