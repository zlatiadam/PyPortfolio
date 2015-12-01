# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 21:25:44 2015

@author: Zlati
"""

from ..base import Portfolio


class EqualWeightsPortfolio(Portfolio):
    
    def __init__(self, df, returns=False, preprocessed=False):
        Portfolio.__init__(self, df, returns, preprocessed)

    
    def rebalance(self, from_date=None, to_date=None):
        n = len(self.__get_returns_df__().columns)
        weights = [1.0/n] * n
        self.__set_weights__(weights, to_date)