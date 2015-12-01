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


def semivariance(data, avg=None, loss=False):
    
    if type(data) == pd.core.frame.DataFrame:
        data = data.values
    
    if avg is None:
        avg = np.mean(data)

    sv_sum = 0.0
    sv_count = 0
    
    for val in data:
        if loss:
            if val >= avg:
                sv_sum += (val - avg)**2
                sv_count += 1
        else:
            if val <= avg:
                sv_sum += (val - avg)**2
                sv_count += 1
           
    return sv_sum / sv_count
    

def semivar(data, avg=None, loss=False):
    return semivariance(data, avg, loss)

    
def value_at_risk(data, alpha, loss=False, interpolation="linear"):
    return np.percentile(data, 1 - alpha, interpolation=interpolation) if loss else -1.0*np.percentile(data, alpha, interpolation=interpolation)


def VaR(data, alpha, loss=False, interpolation="linear"):
    return np.percentile(data, 1 - alpha, interpolation=interpolation) if loss else -1.0*np.percentile(data, alpha, interpolation=interpolation)
    

def drawdown(data, normalize=True, loss=False):
    
    if type(data) == pd.core.frame.DataFrame:
        data = data.values
    
    mdd = 0.0 # max drawdown
    dd = 0.0 # drawdown
    avdd = 0.0 # average drawdown
    
    peak = float("inf") if loss else float("-inf") # may raise error on some CPU-s
    
    for val in data:
        if loss:
            if val < peak:
                peak = val
            dd = (100.0 * (val - peak) / peak) if normalize else (val - peak)
        else:
            if val > peak:
                peak = val
            dd = (100.0 * (peak - val) / peak) if normalize else (peak - val)
        
        if dd > mdd:
            mdd = dd
            
        avdd += dd
            
    return dd, mdd, avdd/len(data)


def DD(data, normalize=True, loss=False):
    return drawdown(data, normalize, loss)[0]


def maxDD(data, normalize=True, loss=False):
    return drawdown(data, normalize, loss)[1]


def avDD(data, normalize=True, loss=False):
    return drawdown(data, normalize, loss)[2]


def conditional_value_at_risk(data, alpha, loss=False):
    if type(data) == pd.core.frame.DataFrame:
        data = data.values

    cvar_sum = 0.0
    cvar_cntr = 0

    VaR_alpha = VaR(data, alpha, loss)
    
    for val in data:
        if loss:
            if val > VaR_alpha:
                cvar_sum += val
                cvar_cntr += 1
        else:
            if val < -1.0*VaR_alpha:
                cvar_sum += val
                cvar_cntr += 1
            
    return cvar_sum / cvar_cntr
    

def CVaR(data, alpha, loss=False):
    return conditional_value_at_risk(data, alpha, loss)
    
    
def ES(data, alpha, loss=False):
    return conditional_value_at_risk(data, alpha, loss)


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