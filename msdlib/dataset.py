"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""


import pandas as pd
import numpy as np


def solenoid_data(data_len=10000, ys=3, delay=.3, z_spread=[0, 50]):
    """
    This function generates 3d data to form solenoidal shape

    Inputs:
        :data_len: int, number of points, default is 10000
        :ys: int, number of solenoids, default is 3
        :delay: float, phase shift parameter for cosine function, default is 0.3
        :z_spread: tuple/array of length 2, containing z-axis starting and ending values stretching all data points. Default is [0, 50]

    Outputs:
        :data_source: pandas DataFrame, contains 'x', 'z' and ys number of y columns (default is 3 for ys, so in total 5 columns to represent 3 solenoidal curves)
    """
    data_source = []
    x = np.linspace(z_spread[0], z_spread[1], data_len)
    z = np.sin(x)
    y = [np.cos(x + i * delay) for i in range(ys)]
    data_source = pd.DataFrame(
        [x, z] + y, index=['z', 'x'] + ['y_%02d' % i for i in range(ys)]).T
    return data_source
