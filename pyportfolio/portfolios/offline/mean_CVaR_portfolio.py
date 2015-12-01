# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 22:00:36 2015

@author: Zlati
"""

import numpy as np

from scipy.optimize import linprog

from ..base import Portfolio


class MeanCVaRPortfolio(Portfolio):
    
    def __init__(self, df, returns=False, preprocessed=False):
        Portfolio.__init__(self, df, returns, preprocessed)
    
    def rebalance(self, alpha, cov_estimator=None, mean_estimator=None, 
                  long_only=True, max_leverage=0.1, min_short=None,
                  max_short=None, min_long=None, max_long=None,
                  from_date=None, to_date=None):

        returns = self.__get_return_matrix__(from_date, to_date)
        T, n = returns.shape

        # min cx
        # Ax <= b
        v = 1 / ( (1 - alpha) * T )
        
        c = np.asarray([1] + [v]*T + [0.0]*n)
        
        A1 = np.append(np.matrix([1.0]*T).T, np.eye(T), axis=1)
        A1 = np.append(A1, returns, axis=1)
        
        A2 = np.append(np.matrix([0.0]*T).T, np.eye(T), axis=1)
        A2 = np.append(A2, np.zeros_like(returns), axis=1)
        
        A3 = np.append(np.matrix([0.0]*n).T, np.zeros((n,T)), axis=1)
        A3 = np.append(A3, np.eye(n), axis=1)
        
        A = np.append(A1, A2, axis=0)
        A = np.append(A, A3, axis=0)

        A_ub = -A
        
        b_ub = np.zeros(2*T+n)
        
        A_eq = [np.append(np.zeros(1+T), np.ones(n), axis=1)]
        b_eq = np.asarray(1.0)
        
        sol = linprog(c, A_ub, b_ub, A_eq, b_eq, options={"disp": True})
        
        weights = list(sol['x'][-n:])
        
        print(sol)
        
        self.__set_weights__(weights, to_date)