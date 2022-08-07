import numpy as np
import pandas as pd
from msdlib import msd
import pickle
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def param_proposal(param, prev_param, score, prev_score, alpha):
    score_grad = alpha * (score - prev_score)
    for p in param:
        param[p] = prev_param[]


params = {
    'x': list(range(1, 100, 10)),
    'y': list(range(0, 15, 2))
}

function = lambda x, y: x**2 - 5*y + 2 - 1/x

optimizer = msd.paramOptimizer(params, mode='grid')
prev_score = None
prev_param = None
while True:
    param = param_proposal(param, prev_param, score, prev_score)
    score = function(param['x'], param['y'])
    print(param, score)
    end = optimizer.set_score(score)
    if end:
        break
    prev_param = param
    prev_score = score

print(optimizer.best())

x = optimizer.queue['x']
y = optimizer.queue['y']
scores = optimizer.queue['score']
plt.figure()
plt.plot(x, y)
plt.scatter(x, y, c=scores, cmap='jet')
plt.show()