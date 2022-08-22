"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""

import os
import sys
import time
import pickle
from msdlib import msd
import numpy as np
import pandas as pd


def test_sample():
    """sample test case to check testing activity"""
    assert True


def test_class_result():
    """
    To check class_result function's activity
    """

    y = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    pred = [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1]

    result_avg_sum = 3.6905
    conf_trace = 9

    result, confmat = msd.class_result(y, pred, True)
    assert result['average'].sum().round(4) == result_avg_sum
    assert confmat.values.trace() == conf_trace


def test_rsquare_rmse():
    """
    To check rsquare_rmse function's activity
    """

    y = [.1, .1, .1, .1, .1, .2, .3, .4, .5, .6, .7]
    pred = [.14, .15, .11, .1, .1, .19, .28, .41, .55, .6, .68]

    rsquare_val = 0.9849
    rmse_val = 0.0265

    rsquare, rmse = msd.rsquare_rmse(y, pred)
    assert rsquare.round(4) == rsquare_val
    assert rmse.round(4) == rmse_val


def test_get_time_estimation():
    """
    To check get_time_estimation function's activity
    """

    expected_timestring = '0:00:02 < 0:00:19'

    time_st = time.time()
    time.sleep(2.2)
    timestring = msd.get_time_estimation(time_st, count_ratio=20/200)
    assert timestring == expected_timestring


def test_paramOptimizer():
    """
    To check paramOptimizer's functionality
    """

    params = {
        'x': [1, 2],
        'y': [3, 4, 5]
    }

    expected_best = -19.5

    def function(x, y): return x**2 - 5*y + 2 - 1/x

    optimizer = msd.paramOptimizer(params, mode='grid')

    while True:
        paramset = optimizer.get_param()
        value = function(paramset['x'], paramset['y'])
        isEnd = optimizer.set_score(value)
        if isEnd:
            break
    best = optimizer.best()[0]['score'].iloc[0]
    assert best == expected_best


def test_get_category_edges():
    """
    To check get_category_edges function
    """
    
    sr = pickle.load(open('tests/data/get_category_edges_sr.pickle', 'rb'))
    categories = np.unique(sr)
    edges = msd.get_category_edges(sr, categories=categories, names=None)
    assert len(edges) == 4 and edges[0]['start'].iloc[0] == 20


def test_grouped_mode():

    data = pd.read_csv('tests/data/US_corn_futures_historical_data.csv')
    data = data.set_index('Date').sort_index()
    mode = msd.grouped_mode(data['Close'], bins=50)
    
    assert mode == 370.3675


def test_feature_evaluator():
    data = pd.read_csv('tests/data/US_corn_futures_historical_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date').sort_index().drop(['Percentage_change', 'Volume'], axis=1)
    data = data.astype(float)

    result = msd.feature_evaluator(data, label_name=['Close'], label_type_num=[True], n_bin=40, is_all_num=True, show=False)

    assert result['Numerical label']['Close_numerics'].iloc[0].idxmax() == 'Low'
    assert result['Numerical label']['Close_numerics']['Low'].iloc[0].round(2) == round(20.852355, 2)


def test_Filters():

    data = pd.read_csv('tests/data/US_corn_futures_historical_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date').sort_index().drop(['Percentage_change', 'Volume'], axis=1)
    data = data.astype(float)

    sr = data['Close'].copy()
    filt_type = 'lp'
    f_cut = .01
    f_lim = [0, .03]
    show = False

    filt = msd.Filters(T=1)
    filt.vis_spectrum(sr, f_lim=f_lim, show=show)
    filt.plot_filter(sr, filt_type=filt_type, f_cut=f_cut, f_lim=f_lim, show=show)
    filt.apply(sr, filt_type, f_cut, order=10, response=True, plot=True, f_lim=f_lim, show=show)
    
    assert True


def test_moving_slope():

    data = pd.read_csv('tests/data/US_corn_futures_historical_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date').sort_index()
    data['slope_Close'] = msd.moving_slope(data[['Close']], win=10, fs='1D')

    assert data['slope_Close'].min() < -20
