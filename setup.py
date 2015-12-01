# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 01:54:01 2015

@author: Zlati
"""

from distutils.core import setup
from setuptools import find_packages

setup(name='pyportfolio',
      version='0.0.1',
      description='Package for online/offline portfolio optimisation',
      url="https://github.com/zlatiadam/PyPortfolio",
      download_url="",
      author="Ádám Zlatniczki",
      author_email="adam.zlatniczki@cs.bme.hu",
      license='GNU General Public License 2.0',
      packages=find_packages(),
      package_data={},
      keywords=["portfolio", "optimisation", "online", "offline", "backtest",
                "robust"],
      install_requires=["pandas", "numpy", "cvxopt", "scipy", "statsmodels",
      "sklearn", "requests"],
      zip_safe=False)