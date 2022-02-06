# 树结点类
class TreeNode(object):
    """
    保存决策树的结点信息：
    1. 划分特征
    2. 划分特征的值
    3. 结点的值（分类是各类别的概率，回归是具体的值）
    4. 结点的深度
    5. 结点样本数量
    6. 子结点
    """

    def __init__(self, best_feature, thres_list, node_value, node_depth, node_samples, child_node=None):
        self.feature_i = best_feature
        self.feature_thres = thres_list
        self.value = node_value
        self.depth = node_depth
        self.n_samples = node_samples
        self.childNode = child_node


# 树生成类
class BaseTree(object):

    def __init__(self, criterion, minloss):
        """
        基础树的类，包括树的构造过程，树的预测过程等等。

        Parameters
        ----------
        criterion : String
            树生成的准则.
        minloss : float
            树停止生长的最小误差，分类树和回归树以及不同的准则会有不同的损失计算函数.

        Returns
        -------
        None

        """
        self.criterion = criterion
        self.minloss = minloss
        self.tree = None
        self._calculate_loss = None  # 结点Loss的计算方法
        self._calculate_nodevalue_method = None  # 结点值的计算方法
        self._search_feature_method = None  # 结点特征选择方法
        self._split_dataset_method = None  # 结点样本拆分方法

    def _build_tree(self, X, y, node_depth=0, column_indicator=None):
        """
        树的构造方法。

        Parameters
        ----------
        X : 2D-Array
            该结点上的样本.
        y : 1D-Array
            该结点上的标签，分类结果或者数值.
        node_depth : Int
            该结点的树深.
        column_indicator : List
            该结点上样本的列的指针.

        Returns
        -------
        node : Class
            该结点的信息，无法再分返回None.

        """
        node_value = self._calculate_nodevalue_method(y)
        n_samples, n_features = X.shape
        # 划分树的话，另child_node为None，条件如下：
        ## 1. 无特征可分，
        ## 2. 分类问题中信息熵很低，或者纯度很高，即loss很小的时候；
        ## 3. 回归问题中loss很小的时候
        loss = self._calculate_loss(y)
        if (n_features == 0) or (loss <= self.minloss):
            node = TreeNode(None, None, node_value, node_depth, n_samples, None)
            return node

        # 不停止划分的话，令child_node为字典：
        child_node = {}
        ## 1.找到树的最优特征和划分方法  
        best_feature, thres_list = self._search_feature_method(X, y, column_indicator)
        ## 2.遍历保存树的子结点
        for thres in thres_list:
            newx, newy, newcolindicator = self._split_dataset_method(X, y, best_feature, thres, column_indicator)
            new_depth = node_depth + 1
            child_node[thres] = self._build_tree(newx, newy, new_depth, newcolindicator)
        node = TreeNode(best_feature, thres_list, node_value, node_depth, n_samples, child_node)
        return node
