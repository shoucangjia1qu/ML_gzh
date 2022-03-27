import numpy as np


# 正向拆解树结构时用到的函数
def tree_split_regressor(X, tree_node, X_index, res_container, is_categorical, value='value'):
    """
    value: str
        defalut 'value', else 'node_id'
    
    """
    # 停止条件：到叶子节点的时候返回值
    if not tree_node.childNode:
        res_container[X_index] = eval('tree_node.{}'.format(value))
        return
    # 继续拆分求解
    for thresname in tree_node.feature_thres:
        if (is_categorical.__len__() > 0 and is_categorical[tree_node.feature_i]==True):
            ## 分类变量的拆分
            if thresname[1] == 'left':
                Xi_position = np.where(X[:, tree_node.feature_i] == thresname[0])[0]
            else:
                Xi_position = np.where(X[:, tree_node.feature_i] != thresname[0])[0]
        else:   
            ## 连续变量的拆分
            if thresname[1] == 'left':
                Xi_position = np.where(X[:, tree_node.feature_i] <= thresname[0])[0]
            else:
                Xi_position = np.where(X[:, tree_node.feature_i] > thresname[0])[0]
        # 有数据集的情况下进行递归求解
        if len(Xi_position) > 0:
            Xi = X.take(Xi_position, axis=0)
            Xi_index = X_index.take(Xi_position, axis=0)
            sub_tree_node = tree_node.childNode[thresname]
            tree_split_regressor(Xi, sub_tree_node, Xi_index, res_container, is_categorical, value)



def tree_split_classifer():
    pass



ANALYSE_TREE = {
    'regressor' : tree_split_regressor,
    'classifer' : tree_split_classifer
    
    }



