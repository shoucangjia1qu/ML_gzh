# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 22:45:25 2022

@author: ecupl
"""
import numpy as np

class LeastSquaresError(object):
    """Loss function for least squares (LS) estimation."""
    
    def __init__(self):
        pass
    
    def __call__(self, y, y_prediction):
        """Compute the least square loss."""
        return np.mean((y - y_prediction.ravel()) ** 2)
    
    def negative_gradient(self, y, y_prediction):
        """Compute the negative gradient."""
        return y - y_prediction.ravel()
    
    def update_terminal_regions(self, tree, terminal_samples_nodeid, X, y, y_prediction):
        pass
        
    def _update_terminal_node(self, tree, terminal_nodes, terminal_leaf, X, y, y_prediction):
        """LS no need to update terminal regions"""
        pass


class LeastAbsoluteError(object):
    """Loss function for least absolute deviation (LAD) regression."""
    
    def __init__(self):
        pass
    
    def __call__(self, y, y_prediction):
        """Compute the least absolute loss."""
        return np.abs(y - y_prediction.ravel()).mean()
    
    def negative_gradient(self, y, y_prediction):
        """Compute the negative gradient."""
        return 2 * ((y - y_prediction.ravel()) > 0) - 1
    
    def update_terminal_regions(self, tree, terminal_samples_nodeid, X, y, y_prediction):
        if not tree.childNode:
            terminal_leaf = tree.node_id
            tree.value = self._update_terminal_node(terminal_samples_nodeid, terminal_leaf, X, y, y_prediction)
            return
        for thres, childTree in tree.childNode:
            self.update_terminal_regions(childTree, terminal_samples_nodeid, X, y, y_prediction)
       
    def _update_terminal_node(self, terminal_samples_nodeid, terminal_leaf, X, y, y_prediction):
        """LS no need to update terminal regions"""
        terminal_region_index = np.where(terminal_samples_nodeid == terminal_leaf)[0]
        diff = (y.take(terminal_region_index, axis=0) - y_prediction.take(terminal_region_index, axis=0))
        return np.median(diff)
        
