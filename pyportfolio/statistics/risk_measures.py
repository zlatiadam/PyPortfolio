# -*- coding: utf-8 -*-
"""
Created on Thu Nov 05 18:12:53 2015

@author: Adam Zlatniczki
"""

import numpy as np
import pandas as pd


def variance(data):
    return np.var(data)


def var(data):
    return np.var(data)

    
def standard_deviation(data):
    return np.std(data)

    
def std(data):
    return np.std(data)


def semivariance(data, avg=None):
    
    if type(data) == pd.core.frame.DataFrame:
        data = data.values
    
    if avg is None:
        avg = np.mean(data)

    sv_sum = 0.0
    sv_count = 0
    
    for val in data:
        if val <= avg:
            sv_sum += (val - avg)**2
            sv_count += 1
           
    return sv_sum / sv_count
    

def semivar(data, avg=None):
    return semivariance(data, avg)

    
def value_at_risk(data, alpha, interpolation="linear"):
    return np.percentile(data, alpha, interpolation=interpolation)


def VaR(data, alpha, interpolation="linear"):
    return np.percentile(data, alpha, interpolation=interpolation)


def drawdown(data, normalize=True):
    
    if type(data) == pd.core.frame.DataFrame:
        data = data.values
    
    mdd = 0.0 # max drawdown
    dd = 0.0 # drawdown
    avdd = 0.0 # average drawdown
    
    peak = float("-inf") # may raise error on some CPU-s
    
    for val in data:
        if val > peak:
            peak = val
        dd = (100.0 * (peak - val) / peak) if normalize else (peak - val)
        
        if dd > mdd:
            mdd = dd
            
        avdd += dd
            
    return dd, mdd, avdd/len(data)


def DD(data, normalize=True):
    return drawdown(data, normalize)[0]


def maxDD(data, normalize=True):
    return drawdown(data, normalize)[1]


def avDD(data, normalize=True):
    return drawdown(data, normalize)[2]


def conditional_value_at_risk(data, alpha):
    if type(data) == pd.core.frame.DataFrame:
        data = data.values

    cvar_sum = 0.0
    cvar_cntr = 0

    VaR_alpha = VaR(data, alpha)
    
    for val in data:
        if val < VaR_alpha:
            cvar_sum += val
            cvar_cntr += 1
            
    return cvar_sum / cvar_cntr
    

def CVaR(data, alpha):
    return conditional_value_at_risk(data, alpha)
    
    
def ES(data, alpha):
    return conditional_value_at_risk(data, alpha)


def leverage(weights):
    equity = 0.0
    debt = 0.0
    
    for weight in weights:
        if weight > 0.0:
            equity += weight
        else:
            debt += -1*weight

    portfolio_leverage = debt / (equity + debt)
    leverage_ratio = debt / equity

    return portfolio_leverage, leverage_ratio
    

def relative_leverage(weights):
    # Debt-to-Equity
    return leverage(weights)[1]
    

def portfolio_leverage(weights):
    return leverage(weights)[0]