# -*- coding: utf-8 -*-
"""
Created on Sat May 23 13:58:22 2020

@author: ecupl
"""

import numpy as np
import pandas as pd

#先统计出每个Device_model对应的点击次数和未点击次数
def click_counting(df, xName, yName):
    click_result = pd.crosstab(df[xName], df[yName])
    click_result['total'] = click_result.sum(axis=1)
    click_result['percent'] = click_result.total / float(df.shape[0])
    return click_result.sort_values(by='percent', ascending=False)


#再进行分箱计数
def bin_countng(counts):
    eps = 0.0001
    counts['N+'] = (counts[1] + eps).divide(counts['total'])
    counts['N-'] = (counts[0] + eps)/counts['total']
    counts['odds'] = counts['N+'].divide(counts['N-'])
    counts['log_odds'] = np.log(counts['N+']) - np.log(counts['N-'])
#    bin_counts = counts.filter(items = ['N+', 'N-', 'odds', 'log_odds'])
    return counts


#处理稀有类
def back_off(counts, thres=5):
    eps = 0.0001
    bf_counts = counts.copy()
    ##计算需要back_off的index，以及正负样本数量
    bf_idx = np.nonzero(bf_counts.total.values<5)[0].copy()
    bf_total = bf_counts['total'].values[bf_idx].sum()
    bf_pos = bf_counts[1].values[bf_idx].sum()
    bf_neg = bf_counts[0].values[bf_idx].sum()
    ##新增一列是否back_off_box，并赋值
    bf_counts['back_off_box'] = 0
    bf_counts['back_off_box'].iloc[bf_idx] = 1
    ##对需要back_off的设备重新赋值
    bf_counts[1].iloc[bf_idx] = bf_pos
    bf_counts[0].iloc[bf_idx] = bf_neg
    bf_counts['total'].iloc[bf_idx] = bf_total
    bf_counts['percent'].iloc[bf_idx] = bf_total / float(df.shape[0])
    bf_counts['N+'].iloc[bf_idx] = (bf_pos + eps) / bf_total
    bf_counts['N-'].iloc[bf_idx] = (bf_neg + eps) / bf_total
    bf_counts['odds'].iloc[bf_idx] = bf_counts['N+'].iloc[bf_idx].divide(bf_counts['N-'].iloc[bf_idx])
    bf_counts['log_odds'].iloc[bf_idx] = np.log(bf_counts['odds'].iloc[bf_idx])
    return bf_counts


if __name__ == "__main__":
    #导入数据
    df = pd.read_csv("train_subset.csv")
    #查看"device_model"分类变量的数值个数
    print(df.device_model.value_counts())
    #分箱计数
    click_counts = click_counting(df, "device_model", "click")
    print(click_counts.head())
    bin_counts = bin_countng(click_counts)
    print(bin_counts.head())
    #处理稀有类
    back_off_bincounts = back_off(bin_counts)
    back_off_bincounts.log_odds.hist(bins=50)
    
    
    
    