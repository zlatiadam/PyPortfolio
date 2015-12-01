# -*- coding: utf-8 -*-
"""
Created on Sun Nov 08 23:55:52 2015

@author: Zlati
"""

import numpy as np
from sklearn.covariance import fast_mcd, ledoit_wolf, oas, empirical_covariance
from statsmodels.robust.scale import huber
from scipy.stats import kendalltau, spearmanr


def hodges_lehmann_mean(x):
    """
    Robust estimator of the mean
    """
    m = np.add.outer(x,x)
    ind = np.tril_indices(len(x), -1)
    return 0.5 * np.median(m[ind])
    
    
def median_absolute_deviation(x):
    """
    Robust estimator for Standard Deviation
    https://en.wikipedia.org/wiki/Robust_measures_of_scale
    """
    x = np.ma.array(x).compressed()
    m = np.median(x)
    return np.median(np.abs(x - m))
    
    
def absolute_pairwise_difference(x, alt="Q"):
    """
    Robust estimators for Standard Deviation
    https://en.wikipedia.org/wiki/Robust_measures_of_scale
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.393.7947&rep=rep1&type=pdf
    """
    if alt == "S":
        c_Sn = 1 if len(x)%2 == 0 else len(x)/(len(x)-0.9)
        return c_Sn*1.1926*np.median([np.median([np.abs(x_i - x_j) for x_j in x]) for x_i in x])
    if alt == "Q":
        c_Qn = len(x)/(len(x)+3.8) if len(x)%2 == 0 else len(x)/(len(x)+1.4)
        a = list()
        for i in range(len(x)):
            for j in range(i, len(x)):
                a.append(np.abs(x[i]-x[j]))
        return c_Qn*np.percentile(a, q=25)

    return None


def biweight_midvariance(x):
    """
    Robust estimator of the variance
    https://en.wikipedia.org/wiki/Robust_measures_of_scale
    """
    mad = median_absolute_deviation(x)
    Q = np.median(x)
    
    num = len(x)*np.sum([(((x_i - Q)**2) * ((1-((x_i-Q)/(9*mad))**2)**4) if np.abs((1-(x_i-Q)/(9*mad))) < 1.0 else 0.0) for x_i in x])
    nom = np.sum([((1-(((x_i-Q)/(9*mad))**2)) * (1-((5*(x_i-Q)/(9*mad))**2)) if np.abs((1-(x_i-Q)/(9*mad))) < 1.0 else 0.0) for x_i in x])**2
    
    return num/nom

###############################################################################

def get_location_estimator(estimator):
    if hasattr(estimator, '__call__'):
        f = estimator
    elif type(estimator) == str:
        if estimator=="HL" or estimator=="hl" or estimator=="hodges_lehmann":
            f = hodges_lehmann_mean
        elif estimator=="median":
            f = np.median
        elif estimator=="Huber" or estimator=="huber":
            f = lambda x: np.asscalar(huber(x)[0])
        else:
            f = np.mean
    else:
        f = np.mean
            
    return f


def get_scale_estimator(estimator):
    if hasattr(estimator, '__call__'):
        f = estimator
    elif type(estimator) == str:
        if estimator=="Huber" or estimator=="huber":
            f = lambda x: np.asscalar(huber(x)[1])
        elif estimator=="MAD" or estimator=="mad" or estimator=="mean_absolute_deviation":
            f = median_absolute_deviation
        elif estimator=="apdS" or estimator=="S":
            f = lambda x: absolute_pairwise_difference(x, "S")
        elif estimator=="apdQ" or estimator=="Q":
            f = lambda x: absolute_pairwise_difference(x, "Q")
        elif estimator=="biweight_midvariance" or estimator=="bmv":
            f = biweight_midvariance
        else:
            f = np.var
    else:
        f = np.var
            
    return f
    
    
def get_covariance_estimator(estimator):
    if hasattr(estimator, '__call__'):
        f = estimator
    elif type(estimator) == str:
        if estimator=="MCD" or estimator=="mcd" or estimator=="MinCovDet" or estimator=="fast_mcd":
            f = fast_mcd
        elif estimator=="Ledoit-Wolf" or estimator=="LW" or estimator=="lw":
            f = lambda x: ledoit_wolf(x)[0]
        elif estimator=="OAS" or estimator=="oas":
            f = lambda x: oas(x)[0]
        else:
            f = empirical_covariance
    else:
        f = empirical_covariance
            
    return f


def get_correlation_estimator(estimator):
    if hasattr(estimator, '__call__'):
        f = estimator
    elif type(estimator) == str:
        if estimator=="kendall_tau":
            raise Exception("Unimplemented!")
        elif estimator=="spearman_rho":
            f = spearmanr
        else:
            f = lambda x: np.corrcoef(x.T)
    else:
        f = lambda x: np.corrcoef(x.T)

    return f