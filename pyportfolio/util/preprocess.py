# -*- coding: utf-8 -*-
"""
Created on Fri Nov 06 10:59:17 2015

@author: Zlati
"""


import pandas as pd
import numpy as np


def convert_prices_to_returns(df):
    return df.diff(periods=1, axis=0)[1:] / np.asarray(df[0:-1])


def preprocess_dataframe(df, returns=False):

    df = df.interpolate("linear", inplace=False)
    
    if "date" in df.columns:
        df.index = pd.to_datetime(df["date"])
        df = df.drop("date", axis=1, inplace=False)
        
    if "Date" in df.columns:
        df.index = pd.to_datetime(df["Date"])
        df = df.drop("Date", axis=1, inplace=False)
    
    if not returns:
        r_f = None
        # convert prices to returns
        if "risk_free" in df.columns:
            r_f = df["risk_free"]
            df = df.drop("risk_free", axis=1)

        df = convert_prices_to_returns(df)
        
        if r_f is not None:
            df["risk_free"] = r_f

    df = df.dropna(axis=1, inplace=False)

    return df