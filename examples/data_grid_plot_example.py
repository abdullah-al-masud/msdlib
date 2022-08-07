"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""

import pandas as pd
from sklearn.datasets import load_iris

import os
import sys
project_dir = os.getcwd()
sys.path.append(project_dir)
from msdlib import msd


# Loading Iris data set from sklearn
loader = load_iris()
print(loader.keys())
feature_names = loader['feature_names']
target_names = loader['target_names']
print(feature_names, target_names)
data = pd.DataFrame(loader['data'], columns=feature_names)

# preparing identifier (idf)
classes = pd.Series(loader['target'], name='Label').replace(
    {i: target_names[i] for i in range(len(target_names))})

print(data)
print(classes)

fig_title = 'Grid plot for Iris dataset'
savepath = 'examples/data_grid_plot_example'
msd.data_gridplot(data, idf=classes, idf_pref='', idf_suff='', diag='kde', figsize=(16, 12), alpha=.7,
                  s=8, lg_font='x-small', lg_loc=1, fig_title=fig_title, show_corr=True, savepath=savepath,
                  show_stat=True, cmap=None, save=True, show=False, fname=fig_title.replace(' ', '_'))
