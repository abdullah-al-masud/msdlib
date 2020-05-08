import pandas as pd
import numpy as np


def solenoid_data(data_len = 10000, ys = 3, delay = .3, z_spread = [0, 50]):
    data_source = []
    x = np.linspace(z_spread[0], z_spread[1], data_len)
    z = np.sin(x)
    y = [np.cos(x + i * delay) for i in range(ys)]
    data_source = pd.DataFrame([x, z] + y, index = ['z', 'x'] + ['y_%02d'%i for i in range(ys)]).T
    return data_source
