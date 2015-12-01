# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 21:57:54 2015

@author: Zlati
"""
import numpy as np
from numpy import linalg

from ..base import Portfolio
from ...statistics import robust_statistics as robust


class SmallestEigenvaluePortfolio(Portfolio):

    def __init__(self, df, returns=False, preprocessed=False):
        Portfolio.__init__(self, df, returns, preprocessed)
        
    
    def rebalance(self, cov_estimator=None, from_date=None, to_date=None):
        cov_estimator = robust.get_covariance_estimator(cov_estimator)

        returns = self.__get_return_matrix__(from_date, to_date)

        cov_matrix = cov_estimator(returns)
        
        evals, evecs = linalg.eig(cov_matrix)
        
        weights = evecs[ np.argmin(evals) ]
        
        equity = sum([weight if weight > 0.0 else 0.0 for weight in weights])
        
        weights = np.asarray(weights) * 1.0 / equity
        
        self.__set_weights__(list(weights), to_date)