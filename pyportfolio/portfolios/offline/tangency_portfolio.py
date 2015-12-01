# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 21:31:02 2015

@author: Zlati
"""

import numpy as np
import pandas as pd
from pandas import datetime as date

from ..base import Portfolio
from pyportfolio.util import yahoo_finance

class TangencyPortfolio(Portfolio):
    
    def __init__(self, r_f, index="^GSPC", from_year="2000", from_month="00", from_day="1", to_year=None, to_month=None, to_day=None, frequency="d"):
        
        if type(index) == str:
            if index not in ["^GSPC", "^DJI", "^IXIC", "^NYA", "^GSPTSE",
                             "^STOXX50E", "^FTSE", "^GDAXI", "^FCHI", "^IBEX",
                             "^N225", "IJIIX", "^HSI", "000300.SS", "^AXJO"]:
                print("The index provided is not in the list of supported indexes - take extra care.")

            to_year = date.today().strftime("%Y") if to_year is None else to_year
            to_month = str(int(date.today().strftime("%m"))-1) if to_month is None else to_month
            to_day = date.today().strftime("%d") if to_day is None else to_day
            
            df = yahoo_finance.get_stock_price_dataframe(index, from_month, from_day, from_year, to_month, to_day, to_year, frequency)
            df = df[["Close"]]
            df.columns = ["market_portfolio"]

        if type(index) == pd.core.frame.DataFrame:
            df = index
        
        if np.isscalar(r_f):
            df["risk_free"] = pd.Series([r_f]*df.index.size, df.index)
        else:
            df["risk_free"] = pd.Series(r_f, df.index)

        Portfolio.__init__(self, df)


    def rebalance(self, l, from_date=None, to_date=None):
        self.__set_weights__([l, 1.0-l], to_date)