# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 21:34:36 2015

@author: Zlati
"""

from .portfolios.offline.equal_weights_portfolio import EqualWeightsPortfolio
from .portfolios.offline.tangency_portfolio import TangencyPortfolio
from .portfolios.offline.minimum_variance_portfolio import MinimumVariancePortfolio
from .portfolios.offline.smallest_eigenvalue_portfolio import SmallestEigenvaluePortfolio
from .portfolios.offline.mean_variance_portfolio import MeanVariancePortfolio
from .portfolios.offline.most_diversified_portfolio import MostDiversifiedPortfolio
from .portfolios.offline.mean_VaR_portfolio import MeanVaRPortfolio
from .portfolios.offline.mean_CVaR_portfolio import MeanCVaRPortfolio

#from pyportfolio.portfolios.base import Portfolio
#df = pd.read_csv("C:\\Users\\Zlati\\Desktop\\PyPortfolio\\pyportfolio\\data\\SP500_daily_2000_2015.csv")