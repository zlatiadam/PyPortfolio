# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 21:57:17 2015

@author: Zlati
"""

import numpy as np

import cvxopt as opt
from cvxopt import solvers

from ..base import Portfolio
from ..constraints import generate_longonly_constraints
from ..constraints import generate_shortallowed_constraints
from ...statistics import robust_statistics as robust


class MostDiversifiedPortfolio(Portfolio):
     
    def __init__(self, df, returns=False, preprocessed=False):
        Portfolio.__init__(self, df, returns, preprocessed)
        
    
    def rebalance(self, corr_estimator=None, std_estimator=None,
                  long_only=True, max_leverage=0.1, min_short=None, max_short=None,
                  min_long=None, max_long=None, from_date=None, to_date=None):

        corr_estimator = robust.get_correlation_estimator(corr_estimator)
        std_estimator = robust.get_scale_estimator(std_estimator)
        
        returns = self.__get_return_matrix__(from_date, to_date)
        n = returns.shape[1]
        
        corr_matrix = corr_estimator(returns)
        
        # volatilities
        stds = np.apply_along_axis(std_estimator, 0, returns)
    
        if long_only:
            P = corr_matrix
            q = np.zeros(n)
            G, h, A, b = generate_longonly_constraints(n, list(self.__get_returns_df__().columns), min_long, max_long)
        else:
            P = np.append(corr_matrix, 0*np.eye(n), axis=1)
            P2 = np.append(0*np.eye(n), corr_matrix, axis=1)
            P = np.append(P, P2, axis=0)
            q = np.zeros(2*n)

            G, h, A, b = generate_shortallowed_constraints(n, list(self.__get_returns_df__().columns), max_leverage, min_short, max_short, min_long, max_long)
        
        P = opt.matrix(P)
        q = opt.matrix(q)
        G = opt.matrix(G)
        h = opt.matrix(h)
        A = opt.matrix(A)
        b = opt.matrix(b)

        
        # Calculate optimal portfolio in the synthetic universe
        weights = list(solvers.qp(P, q, G, h, A, b)["x"])
        
        if not long_only:
            weights = [ weights[i] + weights[len(weights)/2 + i] for i in range(len(weights)/2)]
        
        weights = np.asarray(weights)

        # readjust weights according to the inverse volatilities
        weights = weights / stds

        weights /= sum(weights)

        self.__set_weights__(list(weights), to_date)