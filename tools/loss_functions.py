# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 22:45:25 2022

@author: ecupl
"""
import numpy as np


# 损失函数的基础类
class LossFunction(object):
    def __init__(self):
        pass
    
    def __call__(self, y, y_prediction):
        """
        calculate loss.
        
        Parameters
        ----------
        y : 2D-array
            DESCRIPTION.
        y_prediction : 2D-array
            DESCRIPTION.
        """
        pass
    
    def negative_gradient(self, y, y_prediction, **kwargs):
        """
        calculate negative-gradient.

        Parameters
        ----------
        y : 1D-array
            DESCRIPTION.
        y_prediction : 2D-array
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.
        """
        pass
    
    def update_terminal_regions(self, tree, terminal_samples_nodeid, X, y, y_prediction, **kwargs):
        """update terminal regions"""
        if not tree.childNode:
            terminal_leaf = tree.node_id
            tree.value = self._update_terminal_node(terminal_samples_nodeid, terminal_leaf, X, y, y_prediction)
            return
        for thres, childTree in tree.childNode.items():
            self.update_terminal_regions(childTree, terminal_samples_nodeid, X, y, y_prediction)
    
    def _update_terminal_node(self, tree, terminal_nodes, terminal_leaf, X, y, y_prediction):
        pass


class LeastSquaresError(LossFunction):
    """Loss function for least squares (LS) estimation."""
    
    def __call__(self, y, y_prediction):
        """Compute the least square loss."""
        return np.mean((y.ravel() - y_prediction.ravel()) ** 2)
    
    def negative_gradient(self, y, y_prediction, **kwargs):
        """Compute the negative gradient."""
        return y - y_prediction.ravel()
    
    def update_terminal_regions(self, tree, terminal_samples_nodeid, X, y, y_prediction, **kwargs):
        """LS do not need update terminal regions, so change this function"""
        pass
        

class LeastAbsoluteError(LossFunction):
    """Loss function for least absolute deviation (LAD) regression."""
    
    def __call__(self, y, y_prediction):
        """Compute the least absolute loss."""
        return np.abs(y.ravel() - y_prediction.ravel()).mean()
    
    def negative_gradient(self, y, y_prediction, **kwargs):
        """Compute the negative gradient."""
        return 2 * ((y - y_prediction.ravel()) > 0) - 1
       
    def _update_terminal_node(self, terminal_samples_nodeid, terminal_leaf, X, y, y_prediction):
        terminal_region_index = np.where(terminal_samples_nodeid == terminal_leaf)[0]
        diff = (y.take(terminal_region_index, axis=0) - y_prediction.ravel().take(terminal_region_index, axis=0))
        return np.median(diff)
        

class HuberLossFunction(LossFunction):
    """Huber Loss Function """
    
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.gamma = None
    
    def __call__(self, y, y_prediction):
        """Compute the huber loss."""
        # 先求出区分异常点的值
        diff = y.ravel() - y_prediction.ravel()
        if not self.gamma:
            gamma = np.percentile(np.abs(diff), self.alpha * 100)
        else:
            gamma = self.gamma
        normal_partern = np.abs(diff) <= gamma
        square_loss = np.sum(0.5 * diff[normal_partern] ** 2)
        abs_loss = np.sum(gamma * (np.abs(diff[~normal_partern]) - gamma / 2))
        loss = (square_loss + abs_loss) / y.shape[0]
        return loss
    
    def negative_gradient(self, y, y_prediction, **kwargs):
        """Compute the negative gradient."""
        diff = y-y_prediction.ravel()
        gamma = np.percentile(np.abs(diff), self.alpha * 100)
        normal_partern = np.abs(diff) <= gamma
        negative_gra = np.zeros(y.shape)
        negative_gra[normal_partern] = diff[normal_partern]
        negative_gra[~normal_partern] = gamma * np.sign(diff[~normal_partern])
        self.gamma = gamma
        return negative_gra
       
    def _update_terminal_node(self, terminal_samples_nodeid, terminal_leaf, X, y, y_prediction):
        gamma = self.gamma
        terminal_region_index = np.where(terminal_samples_nodeid == terminal_leaf)[0]
        diff = (y.take(terminal_region_index, axis=0) - y_prediction.ravel().take(terminal_region_index, axis=0))
        median_diff = np.median(diff)
        terminal_region_value = median_diff + np.mean(
            np.sign(diff-median_diff)*
            np.minimum(gamma, abs(diff-median_diff))
            )
        return terminal_region_value


class BinomialLog(LossFunction):
    """Binomial Log-Likelihood Loss Function """
        
    def __call__(self, y, y_prediction):
        return -1 * np.mean((y.ravel() * y_prediction.ravel()) - np.logaddexp(0, y_prediction.ravel()))
    
    def negative_gradient(self, y, y_prediction, **kwargs):
        """Compute the negative gradient."""
        return y - 1 / (1 + np.exp(-1*y_prediction.ravel()))
       
    def _update_terminal_node(self, terminal_samples_nodeid, terminal_leaf, X, y, y_prediction):
        """residual = y - y_prediction
            value = sum(y - prob) / sum(prob * (1 - prob))
            value = sum(residual) / sum((y - residual) * (1 -y + residual))
        """
        terminal_region_index = np.where(terminal_samples_nodeid == terminal_leaf)[0]
        yi = y.take(terminal_region_index, axis=0) 
        yi_prediction = y_prediction.ravel().take(terminal_region_index, axis=0)
        yi_residual = self.negative_gradient(yi, yi_prediction)
        terminal_region_value = np.sum(yi_residual) / np.sum((yi - yi_residual) * (1 - yi + yi_residual))
        return terminal_region_value
    
    def _raw_predict_proba(self, y_prediction):
        proba = np.ones((y_prediction.shape[0], 2), dtype=np.float64)
        proba[:, 1] = 1 / (1 + np.exp(-1*y_prediction.ravel()))
        proba[:, 0] -= proba[:, 1]
        return proba
    
    def _raw_predict_label(self, y_prediction):
        proba = self._raw_predict_proba(y_prediction)
        return np.argmax(proba, axis=1)

    
    
        
class MultinomialLog(LossFunction):
    """Multinomial Log-Likelihood Loss Function """
    
    def __init__(self, K):
        self.K = K
    
    def __call__(self, y, y_prediction):
        """
        y : (n_samples, n_classes)
        y_prediction : (n_samples, n_classes)

        """
        return np.mean(-1 * (y * y_prediction).sum(axis=1) + np.log(np.exp(y_prediction).sum(axis=1)))
    
    def negative_gradient(self, y, y_prediction, k=0, **kwargs):
        """Compute the negative gradient."""
        return y - np.exp(y_prediction[:,k]) / np.exp(y_prediction).sum(axis=1)
       
    def update_terminal_regions(self, tree, terminal_samples_nodeid, X, y, y_prediction, k=0, **kwargs):
        """update terminal regions"""
        if not tree.childNode:
            terminal_leaf = tree.node_id
            tree.value = self._update_terminal_node(terminal_samples_nodeid, terminal_leaf, X, y, y_prediction, k=k)
            return
        for thres, childTree in tree.childNode.items():
            self.update_terminal_regions(childTree, terminal_samples_nodeid, X, y, y_prediction, k=k)
    
    def _update_terminal_node(self, terminal_samples_nodeid, terminal_leaf, X, y, y_prediction, k=0):
        """residual = y - y_prediction
            v1 = (K-1)/K
            value = (v1) * (sum(residual) / sum((y - residual) * (1 -y + residual)))
        """
        terminal_region_index = np.where(terminal_samples_nodeid == terminal_leaf)[0]
        yi = y.take(terminal_region_index, axis=0) 
        yi_prediction = y_prediction.take(terminal_region_index, axis=0)
        yi_residual = self.negative_gradient(yi, yi_prediction, k)
        v1 = (self.K - 1) / self.K
        terminal_region_value = v1 * np.sum(yi_residual) / np.sum((yi - yi_residual) * (1 - yi + yi_residual))
        return terminal_region_value
    
    def _raw_predict_proba(self, y_prediction):
        proba = np.zeros(y_prediction.shape)
        for i in range(y_prediction.shape[1]):
            proba[:, i] = np.exp(y_prediction[:,i]) / np.exp(y_prediction).sum(axis=1)
        return proba
    
    def _raw_predict_label(self, y_prediction):
        proba = self._raw_predict_proba(y_prediction)
        return np.argmax(proba, axis=1)
