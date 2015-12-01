# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 21:59:21 2015

@author: Zlati
"""

import numpy as np
from scipy.stats import norm

import cvxopt as opt
from cvxopt import solvers

from ..base import Portfolio
from ..constraints import generate_longonly_constraints
from ..constraints import generate_shortallowed_constraints
from ...statistics import robust_statistics as robust


class MeanVaRPortfolio(Portfolio):
    
    def __init__(self, df, returns=False, preprocessed=False):
        Portfolio.__init__(self, df, returns, preprocessed)


    def rebalance(self, alpha, cov_estimator=None, mean_estimator=None, 
                  long_only=True, max_leverage=0.1, min_short=None,
                  max_short=None, min_long=None, max_long=None,
                  from_date=None, to_date=None):
        mean_estimator = robust.get_location_estimator(mean_estimator)
        cov_estimator = robust.get_covariance_estimator(cov_estimator)
        
        returns = self.__get_return_matrix__(from_date, to_date)
        n = returns.shape[1]
        
        exp_values = np.apply_along_axis(mean_estimator, 0, returns)
        exp_values = np.asarray(exp_values)

        z = norm.ppf(1 - alpha) # these are returns, not losses

        cov_matrix = cov_estimator(returns)
        
        if long_only:
            P = z*cov_matrix
            q = -exp_values
            G, h, A, b = generate_longonly_constraints(n, list(self.__get_returns_df__().columns), min_long, max_long)
        else:
            P = np.append(z*cov_matrix, 0*np.eye(n), axis=1)
            P2 = np.append(0*np.eye(n), z*cov_matrix, axis=1)
            P = np.append(P, P2, axis=0)
            q = np.append(-exp_values, -exp_values, axis=1)

            G, h, A, b = generate_shortallowed_constraints(n, list(self.__get_returns_df__().columns), max_leverage, min_short, max_short, min_long, max_long)
        
        P = opt.matrix(P)
        q = opt.matrix(q)
        G = opt.matrix(G)
        h = opt.matrix(h)
        A = opt.matrix(A)
        b = opt.matrix(b)
        
        # Calculate optimal portfolio
        weights = list(solvers.qp(P, q, G, h, A, b)["x"])
        
        if not long_only:
            weights = [ weights[i] + weights[len(weights)/2 + i] for i in range(len(weights)/2) ]
        
        self.__set_weights__(weights, to_date)