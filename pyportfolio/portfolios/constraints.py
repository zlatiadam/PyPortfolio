# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 14:06:35 2015

@author: Zlati
"""

import numpy as np


def generate_longonly_constraints(n, columns, min_weights=None, max_weights=None):
    # Gx >= h constraints, in the form of Gx <= h
    G = -np.eye(n)
    
    if type(min_weights) == list:
        h = -np.asarray(min_weights)
    elif type(min_weights) == dict:
        h = np.zeros(n)
        for k,v in min_weights.iteritems():
            h[ columns.index(k) ] = -1*v
    elif np.isscalar(min_weights):
        h = np.asarray([min_weights]*n)
    else:
        h = np.zeros(n)
    
    # Gx <= h constraints
    if max_weights is not None:
        G = np.append(G, np.eye(n), 0)        
        if type(max_weights) == list:
            h2 = np.asarray(max_weights)
        elif type(max_weights) == dict:
            h2 = np.ones(n)
            for k,v in max_weights.iteritems():
                h2[ columns.index(k) ] = v
        elif np.isscalar(max_weights):
            h2 = np.asarray([max_weights]*n)
        else:
            h2 = np.ones(n)
        h = np.append(h, h2, 1)
    
    # Ax = b constraint
    A = np.asmatrix(np.ones(n))
    b = 1.0
    
    return G, h, A, b
    

def generate_shortallowed_constraints(n, columns, max_leverage=0.1,
                                      min_short=None, max_short=None,
                                      min_long=None, max_long=None):

    x = max_leverage / (1 - max_leverage)

    # Gx <= h

    # Long-part
    G = np.append(-np.eye(n), 0*np.eye(n), axis=1) # Y

    if type(min_long) == list:
        h = np.asarray(min_long)
    elif type(min_long) == dict:
        h = np.zeros(n)
        for k,v in min_long.iteritems():
            h[ columns.index(k) ] = v
    elif np.isscalar(min_long):
        h = np.asarray([min_long]*n)
    else:
        h = np.zeros(n)
        
    if max_long is not None:
        G = np.append(G, np.append(np.eye(n), 0*np.eye(n), axis=1), 0)
        if type(max_long) == list:
            h2 = np.asarray(max_long)
        elif type(max_long) == dict:
            h2 = np.ones(n)
            for k,v in max_long.iteritems():
                h2[ columns.index(k) ] = v
        elif np.isscalar(max_long):
            h2 = np.asarray([max_long]*n)
        else:
            h2 = np.ones(n)
        h = np.append(h, h2, 1)
     
    # Short-part
    G = np.append(G, np.append(0*np.eye(n), np.eye(n), axis=1), axis=0) # Z
   
    if type(min_short) == list:
        h2 = np.asarray(min_short)
    elif type(min_short) == dict:
        h2 = np.zeros(n)
        for k,v in min_short.iteritems():
            h2[ columns.index(k) ] = v
    elif np.isscalar(min_short):
        h2 = np.asarray([min_short]*n)
    else:
        h2 = np.zeros(n)
    h = np.append(h, -h2, 1) # shorting has to be converted to negative
        
    if max_short is not None:
        G = np.append(G, np.append(0*np.eye(n), np.eye(n), axis=1), 0)
        if type(max_short) == list:
            h2 = np.asarray([min(i, x) for i in max_short]) # weight must not be greater than x, the allowed leverage
        elif type(max_short) == dict:
            h2 = np.asarray([x]*n)
            for k,v in max_long.iteritems():
                h2[ columns.index(k) ] = min(v, x)
        elif np.isscalar(max_short):
            h2 = np.asarray([max_short]*n)
        else:
            h2 = np.asarray([x]*n)
        h = np.append(h, -h2, 1)
    
    G = np.append(G, [np.append(np.zeros(n), -np.ones(n), axis=1)], axis=0) # sum Z <= x
    h = np.append(h, x)
    
    # Ax = b
    A = np.asmatrix(np.append(np.ones(n), np.zeros(n), axis=1)) # sum Y = 1
    b = 1.0

    return G, h, A, b