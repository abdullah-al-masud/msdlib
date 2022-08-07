import warnings
import matplotlib.pyplot as plt
import os
import sys
project_dir = os.getcwd()
sys.path.append(project_dir)
from msdlib import msd

# warnings.filterwarnings('ignore')


params = {
    'x': list(range(1, 100, 10)),
    'y': list(range(0, 15, 2))
}

function = lambda x, y: x**2 - 5*y + 2 - 1/x

######### GRID search #####################
optimizer = msd.paramOptimizer(params, mode='grid')
while True:
    param = optimizer.get_param()
    score = function(param['x'], param['y'])
    end = optimizer.set_score(score)
    if end:
        break

print(optimizer.best())


######### RANDOM search #####################
optimizer = msd.paramOptimizer(params, mode='random', iteration=15)
while True:
    param = optimizer.get_param()
    score = function(param['x'], param['y'])
    end = optimizer.set_score(score)
    if end:
        break

print(optimizer.best())
