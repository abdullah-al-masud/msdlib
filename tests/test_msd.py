"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""

from msdlib import msd
import time


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


def get_time_estimation():
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
