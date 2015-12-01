# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 21:24:17 2015

@author: Zlati
"""

import numpy as np
import pandas as pd

from pyportfolio.util.preprocess import preprocess_dataframe
from pyportfolio.statistics import risk_measures


class Portfolio:

    def __init__(self, df, returns=False, preprocessed=False):
        """Initialize a portfolio with historical data.
        :param df: either raw prices or returns, preferably in Pandas DataFrame
        :param returns: Indicates whether the provided data contains returns or prices
        :param preprocessed: Indicates whether the dataframe is cleaned: columns must contain only returns/prices, without missing values; index must be Timestamp-type
        """
        self.__weight_history = list()

        if not preprocessed:
            self.__returns_df= preprocess_dataframe(df, returns)
        else:
            self.__returns_df = df


    def __set_weights__(self, weights, at_date):
        if at_date is None:
            at_date = self.__returns_df.index[-1]

        if type(at_date) == str:
            at_date = pd.to_datetime(at_date)
            
        if type(at_date) == int:
            at_date = self.__returns_df.index[at_date]

        for i in xrange(len(self.__weight_history)):
            k = self.__weight_history[i][0]
            if k > at_date:
                return
            
            if at_date == k:
                self.__weight_history[i] = (at_date, weights)
                return

        self.__weight_history.append((at_date, weights))
        self.__weight_history = sorted(self.__weight_history, key=lambda (x,y): x)


    def __set_returns_df__(self, returns_df):
        self.__returns_df = returns_df


    def __get_returns_df__(self, from_date=None, to_date=None):
        if from_date is None:
            from_date = self.__returns_df.index[0]

        if to_date is None:
            to_date = max(self.__returns_df.index)
            
        if type(from_date) == str:
            from_date = pd.to_datetime(from_date)
        
        if type(to_date) == str:
            to_date = pd.to_datetime(to_date)
            
        if type(from_date) == int:
            from_date = self.__returns_df.index[from_date]
            
        if type(to_date) == int:
            to_date = self.__returns_df.index[to_date]
        
        selector = self.__returns_df.index.map(lambda x: from_date <= x <= to_date)
        
        return self.__returns_df[selector]


    def __get_weights_at__(self, at_date=None):

        if len(self.__weight_history) == 0:
            raise Exception("The portfolio weights must be calculated first!")

        if type(at_date) == str:
            at_date = pd.to_datetime(at_date)
            
        if type(at_date) == int:
            at_date = self.__returns_df.index[at_date]
        
        if at_date:
            if at_date < self.__weight_history[0][0]:
                return np.full((1, self.__returns_df.shape[1]), np.nan)
                
            if at_date >= self.__weight_history[-1][0]:
                return self.__weight_history[-1][1]
            
            for i in xrange(len(self.__weight_history)-1):
                lower = self.__weight_history[i][0]
                upper = self.__weight_history[i+1][0]
                
                if lower <= at_date < upper:
                    return self.__weight_history[i][1]
        else:
            return self.__weight_history[-1][1]
        
        
    def __get_weights_in_range__(self, rebalanced=True, from_date=None, to_date=None):
        if from_date is None:
            from_date = self.__returns_df.index[0]

        if to_date is None:
            to_date = self.__returns_df.index[-1]
            
        if type(from_date) == str:
            from_date = pd.to_datetime(from_date)
        
        if type(to_date) == str:
            to_date = pd.to_datetime(to_date)
            
        if type(from_date) == int:
            from_date = self.__returns_df.index[from_date]
            
        if type(to_date) == int:
            to_date = self.__returns_df.index[to_date]
        
        indices = map(lambda x: from_date <= x <= to_date, self.__returns_df.index)
        
        date_indices = self.__returns_df.index[indices]
        
        dates = [None] * len(date_indices)
        weights = [None] * len(date_indices)
        i = 0
        
        for at_date in date_indices:
            dates[i] = at_date
            weights[i] = self.__get_weights_at__(at_date)
            i += 1

        return dates, weights


    def __get_return_matrix__(self, from_date=None, to_date=None):
        if from_date is None:
            from_date = self.__returns_df.index[0]

        if to_date is None:
            to_date = max(self.__returns_df.index)
            
        if type(from_date) == str:
            from_date = pd.to_datetime(from_date)
        
        if type(to_date) == str:
            to_date = pd.to_datetime(to_date)
        
        if type(from_date) == int:
            from_date = self.__returns_df.index[from_date]
            
        if type(to_date) == int:
            to_date = self.__returns_df.index[to_date]
        
        selector = self.__returns_df.index.map(lambda x: from_date <= x <= to_date)
        returns = np.asmatrix(self.__returns_df[selector])
        
        return returns


    def reset_weights(self):
        self.__weight_history = list()
        
    
    def truncate_weights(self, epsilon):
        for i in range(len(self.__weight_history)):
            weights = np.asarray(map(lambda x: x if abs(x) >= epsilon else 0, self.__weight_history[i][1]))
            weights /= np.sum(np.abs(weights))
            self.__weight_history[i][1] = weights


    def rebalance(self, from_date=None, to_date=None):
        raise Exception("The rebalancing method must be overwritten for each different portfolio!")


    def periodic_rebalance(self, frequency, start, time_window=None, kwargs=None):
        if type(start) == str:
            start = pd.to_datetime(start)
            
        if type(start) == pd.tslib.Timestamp:
            start = list(self.__returns_df.index).index(start)
            
        for i in xrange(start + frequency, len(self.__returns_df.index), frequency):
            to_date = self.__returns_df.index[i]
            if time_window:
                from_date = self.__returns_df.index[max(0, i - time_window)]
            else:
                from_date = self.__returns_df.index[0]

            if kwargs is None:
                self.rebalance(from_date=from_date, to_date=to_date)
            else:
                kwargs["from_date"] = from_date
                kwargs["to_date"] = to_date
                self.rebalance(**kwargs)


    def returns(self, rebalanced=True, from_date=None, to_date=None, as_df=True, dropNA=True):
        returns = self.__get_return_matrix__(from_date, to_date)
        
        dates, weights = self.__get_weights_in_range__(rebalanced, from_date, to_date)
        
        portfolio_returns = list()
        
        for i in xrange(0, len(weights)):
            portfolio_returns.append(np.asscalar(np.asarray(weights[i]).dot(returns[i].T)))
        
        if as_df:
            dataframe = pd.DataFrame(portfolio_returns, dates, ["portfolio_return"])
            
            if dropNA:
                dataframe.dropna(inplace=True)
        
        return dataframe if as_df else portfolio_returns

    
    def total_return(self, window=0, rebalanced=True, from_date=None, to_date=None):
        ret = None
        returns = self.returns(rebalanced, from_date, to_date) + 1

        if window == 0:
            ret = np.asscalar(np.prod(returns))
        if window > 0:
            ret = pd.rolling_apply(returns, window, np.prod)
        if window == -1:
            ret = pd.expanding_apply(returns, np.prod)
        
        return  ret


    def expected_value(self, window=0, rebalanced=True, from_date=None, to_date=None):
        ret = None
        returns = self.returns(rebalanced, from_date, to_date)

        if window == 0:
            ret = np.asscalar(np.mean(returns))
        if window > 0:
            ret = pd.rolling_mean(returns, window)
        if window == -1:
            ret = pd.expanding_mean(returns)
        
        return ret
        

    def variance(self, window=0, rebalanced=True, from_date=None, to_date=None):
        ret = None
        returns = self.returns(rebalanced, from_date, to_date)
        
        if window == 0:
            ret = np.asscalar(np.var(returns))
        if window > 0:
            ret = pd.rolling_var(returns, window)
        if window == -1:
            ret = pd.expanding_var(returns)
        
        return ret


    def std(self, window=0, rebalanced=True, from_date=None, to_date=None):
        ret = None
        returns = self.returns(rebalanced, from_date, to_date)
        
        if window == 0:
            ret = np.asscalar(np.std(returns))
        if window > 0:
            ret = pd.rolling_std(returns, window)
        if window == -1:
            ret = pd.expanding_std(returns)

        return ret
        
        
    def semivariance(self, window=0, rebalanced=True, from_date=None, to_date=None):
        ret = None
        returns = self.returns(rebalanced, from_date, to_date)
        
        if window == 0:
            ret = np.asscalar(risk_measures.semivariance(returns))
        if window > 0:
            ret = pd.rolling_apply(returns, window, risk_measures.semivariance)
        if window == -1:
            ret = pd.expanding_apply(returns, risk_measures.semivariance)

        return  ret


    def VaR(self, alpha, window=0, rebalanced=True, from_date=None, to_date=None):
        ret = None
        returns = self.returns(rebalanced, from_date, to_date)
        
        if window == 0:
            ret = np.asscalar(risk_measures.VaR(returns, alpha))
        if window > 0:
            ret = pd.rolling_apply(returns, window, risk_measures.VaR, kwargs={"alpha": alpha})
        if window == -1:
            ret = pd.expanding_apply(returns, risk_measures.VaR, kwargs={"alpha": alpha})
        
        return ret


    def CVaR(self, alpha, window=0, rebalanced=True, from_date=None, to_date=None):
        ret = None
        returns = self.returns(rebalanced, from_date, to_date)
        
        if window == 0:
            ret = np.asscalar(risk_measures.CVaR(returns, alpha))
        if window > 0:
            ret = pd.rolling_apply(returns, window, risk_measures.CVaR, kwargs={"alpha": alpha})
        if window == -1:
            ret = pd.expanding_apply(returns, risk_measures.CVaR, kwargs={"alpha": alpha})
            
        return ret


    def DD(self, normalize=True, window=0, rebalanced=True, from_date=None, to_date=None):
        ret = None
        returns = self.total_return(-1, rebalanced, from_date, to_date)
        
        if window == 0:
            ret = np.asscalar(risk_measures.DD(returns, normalize))
        if window > 0:
            ret = pd.rolling_apply(returns, window, risk_measures.DD, kwargs={"normalize": normalize})
        if window == -1:
            ret = pd.expanding_apply(returns, risk_measures.DD, kwargs={"normalize": normalize})
        
        return ret


    def maxDD(self, normalize=True, window=0, rebalanced=True, from_date=None, to_date=None):
        ret = None
        returns = self.total_return(-1, rebalanced, from_date, to_date)
        
        if window == 0:
            ret = np.asscalar(risk_measures.maxDD(returns, normalize))
        if window > 0:
            ret = pd.rolling_apply(returns, window, risk_measures.maxDD, kwargs={"normalize": normalize})
        if window == -1:
            ret = pd.expanding_apply(returns, risk_measures.maxDD, kwargs={"normalize": normalize})
        
        return ret


    def value(self, init_cash=1.0, window=0, rebalanced=True, from_date=None, to_date=None):
        df = init_cash * self.total_return(window, rebalanced, from_date, to_date)
        df.columns = ["portfolio_value"]
        return df


    def efficient_frontier(self):
        raise Exception("The efficient frontier calculation must be overwritten for each different portfolio!")


    def leverage(self, measure="PL", rebalanced=True, from_date=None, to_date=None):
        leverage = None
        
        if rebalanced:
            leverage = []
            
            index = self.__get_returns_df__(from_date, to_date).index
            
            for timestamp in index:
                weights = weights = self.__get_weights_at__(timestamp)
                
                if measure == "RL":
                    leverage.append(risk_measures.relative_leverage(weights))
                elif measure == "PL":
                    leverage.append(risk_measures.portfolio_leverage(weights))
                else:
                    leverage.append(risk_measures.portfolio_leverage(weights))
            
            leverage = pd.Series(leverage, index)
        else:
            weights = self.__get_weights_at__()
            if measure == "RL":
                leverage = risk_measures.relative_leverage(weights)
            elif measure == "PL":
                leverage =  risk_measures.portfolio_leverage(weights)
            else:
                leverage =  risk_measures.portfolio_leverage(weights)
            
        return leverage