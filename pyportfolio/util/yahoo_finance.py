# -*- coding: utf-8 -*-
"""
Created on Thu Nov 05 19:52:54 2015

@author: Zlati
"""

import requests
from pandas import DataFrame
from io import StringIO

def get_stock_price_dataframe(s, a, b, c, d, e, f, g):
    
    url = 'http://real-chart.finance.yahoo.com/table.csv?'
    
    url += 's=' + s
    url += '&a=' + a
    url += '&b=' + b
    url += '&c=' + c
    url += '&d=' + d
    url += '&e=' + e
    url += '&f=' + f
    url += '&g=' + g
    
    url += '&ignore=.csv'
    
    csv = requests.get(url)
    
    if csv.ok:
        return DataFrame.from_csv(StringIO(csv.text), sep=',').sort()
    else:
        return None
        

def get(s, a, b, c, d, e, f, g):
    return get_stock_price_dataframe(s, a, b, c, d, e, f, g)