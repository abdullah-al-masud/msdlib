"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""


import numpy as np
import pandas as pd


def likelihood_for_categorics(x, uniques=None):
    """
    This function calculates likelihood for variable x for each categorical value in it.
    
    Here, x is Discrete Random Variable.
    For example, we get x from Titanic Dataset for gender variable if we take gender values of Class Survived=1 (so, subset of all gender values). So, this function will provide likelihood of Class Survived=1 (people who survived) for gender variable.
    
    If x doesnt contain all possible values of this variable, then we must provide all possible values in uniques variable.
    
    Inputs:
        :x: Pandas Series or numpy 1-D array, contains discrete random variable for a particular class.
        :uniques: list, pandas Series or numpy array, containing all possible values of broader/parent set of x. Default if None which assumes that x contains all possible values of parent set.
        
    Outputs:
        :probs: pandas Series, it contains likelihood estimations of x for each discrete value. The indices of probs show discrete values of x.
    """
    
    probs = x.value_counts() / x.shape[0]
    if uniques is not None:
        for val in uniques:
            if val not in probs.index:
                probs.loc[val] = 0
        prob = probs.sort_index()
    return probs


def likelihood_for_numerics(x, _min=None, _max=None, bins=100):
    """
    This function converts continues distribution into discrete distribution and calculates likelihood for each distribution bin.
    
    Here, x is Continuous Random Variable.
    For example, we get x from Iris Dataset for petal length variable if we take petal lengths of Class 'Setosa'. So, this function will provide likelihood of 'Setosa' for Petal length.
    
    If the minimum and maximum values of complete distrubition (all petal lengths) is out of bound of the distribution of x, then _min and _max must be provided.
    
    Inputs:
        :x: Pandas Series or numpy 1-D array, contains random variable for a particular class.
        :_min: float, minimum value of the whole distribution where all classes are included. Default is None which assumes the minimum value is minimum of x.
        :_max: float, maximum value of the whole distribution where all classes are included. Default is None which assumes the maximum value is maximum of x.
        :bins: int, number of distribution bins, used to convert continuous distribution into discrete distribution. Default is 100.
    
    Outputs:
        :probs: pandas Series, it contains likelihood estimations of x for each bin. The indices of probs show the upper range/cut of each bin.
    """
    
    if len(x) < bins:
        bins = len(x)
    if _min is None:
        _min = np.min(x)
    if _max is None:
        _max = np.max(x)
    
    _min -= 1e-8
    ranges = np.arange(bins+1) / (bins) * (_max - _min) + _min
    probs = []
    for i in range(len(ranges) - 1):
        probs.append(((x > ranges[i]) & (x <= ranges[i+1])).sum() / x.shape[0])
    probs = pd.Series(probs, index=ranges[1:], name=x.name if isinstance(x, pd.Series) else None)
    
    return probs