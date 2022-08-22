"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""

import numpy as np
import pandas as pd
from msdlib import msd
import pickle
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class RealTimePlotter():

    def __init__(self, params=[], titles=[], xlabels=[], ylabels=[], nrow=None, fig_title=None,
                 savepath=None, max_window=None, figsize=(10, 7), interval=.1, **kwargs):

        self.xlabels = xlabels
        self.ylabels = ylabels
        self.titles = titles
        self.savepath = savepath
        self.max_window = max_window
        self.figsize = figsize
        self.fig_title = fig_title if fig_title is not None else 'Real Time plotting'
        self.nrow = nrow
        self.interval = int(interval * 1000)

        self.colors = get_named_colors()
        self.organize_params(params)
        self.organize_others()
        self.initialize_figure()

    def organize_params(self, params):
        self.params = params
        self.total_params = len(params)
        self.param2index = {}
        self.param2color = {}
        self.param_values = {}
        for i, param in enumerate(self.params):
            if isinstance(param, list):
                for j, p in enumerate(param):
                    self.param2index[p] = i
                    self.param2color[p] = self.colors[j]
                    self.param_values[p] = []
            else:
                self.param2index[param] = i
                self.param2color[param] = self.colors[0]
                self.param_values[param] = []
        self.index2param = {self.param2index[p]: p for p in self.param2index}

    def organize_others(self):
        if self.nrow is None:
            self.nrow = int(np.ceil(np.sqrt(self.total_params)))
        self.ncol = int(self.total_params // self.nrow + int(bool(self.total_params % self.nrow)))
        self.ncol = 1 if self.ncol < 1 else self.ncol

        if len(self.xlabels) < self.total_params:
            for i in range(len(self.xlabels), self.total_params):
                self.xlabels.append('Count')
        if len(self.ylabels) < self.total_params:
            for i in range(len(self.ylabels), self.total_params):
                self.ylabels.append('Values')
        if len(self.titles) < self.total_params:
            for i in range(len(self.titles), self.total_params):
                self.titles.append('Param')

    def initialize_figure(self):
        self.fig = plt.figure(figsize=self.figsize)
        self.fig.suptitle(self.fig_title)
        self.anim = FuncAnimation(fig=self.fig, func=self.draw_plot, interval=self.interval, repeat=False, blit=False)
        plt.show()
        if self.savepath is not None:
            self.anim.save(os.path.join(self.savepath, self.fig_title))

    def set_params(self, param):
        for p in param:
            self.param_values[p].append(param[p])
            if self.max_window is not None:
                self.param_values[p] = self.param_values[p][:-window]
    
    def draw_plot(self, _, q):
        
        plt.clf()
        self.ax = []
        for r in range(self.nrow):
            for c in range(self.ncol):
                i = r * self.ncol + c
                if i < self.total_params:
                    self.ax.append(self.fig.add_subplot(self.nrow, self.ncol, i + 1))
                    self.ax[i].set_title(self.titles[i])
                    self.ax[i].set_xlabel(self.xlabels[i])
                    self.ax[i].set_ylabel(self.ylabels[i])
                else:
                    break
        for p in self.params:
            i = self.param2index[p]
            self.ax[i].plot(self.param_values[p], color=self.param2color[p])
        
        for param in self.params:
            if isinstance(param, list):
                lg_handles = []
                for i, p in enumerate(param):
                    lg = mlines.Line2D([], [], color=self.param2color[p], label=p)
                    lg_landles.append(lg)
                self.ax[self.param2index[p]].legend(handles=lg_handles)
            else:
                lg_handles = [mlines.Line2D([], [], color=self.param2color[param], label=param)]
                self.ax[self.param2index[param]].legend(handles=lg_handles, loc='upper left')
        self.fig.tight_layout()


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