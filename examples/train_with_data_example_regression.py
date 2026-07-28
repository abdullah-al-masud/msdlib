"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""

from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.tree import DecisionTreeRegressor as DTR
import pandas as pd
import torch

import os
import sys
project_dir = os.getcwd()
sys.path.append(project_dir)
from tests.bundled_datasets import load_california_housing_frames
from msdlib import mlutils
from msdlib import msd


savepath = 'examples/train_with_data_regression'

# Loading the data and separating data and label
data, label = load_california_housing_frames()
feature_names = list(data.columns)

# print(source_data['DESCR'])
print('data:\n', data.head())

# Standardizing numerical data
data = msd.standardize(data)

# Splitting data set into train, validation and test
splitter = msd.SplitDataset(data, label, test_ratio=.1)
outdata = splitter.random_split(val_ratio=.1)

print('outdata.keys() :', outdata.keys())
print("outdata['train'].keys() :", outdata['train'].keys())
print("outdata['validation'].keys() :", outdata['validation'].keys())
print("outdata['test'].keys() :", outdata['test'].keys())
print('train > data, labels and index shapes :',
      outdata['train']['data'].shape, outdata['train']['label'].shape, outdata['train']['index'].shape)

# defining layers inside a list
layers = mlutils.define_layers(data.shape[1], 1, [100, 100, 100, 100, 100, 100], dropout_rate=.2, model_type='regressor',
                               actual_units=True, activation=torch.nn.ReLU())

# building model
tmodel = mlutils.torchModel(layers=layers, model_type='regressor',
                            savepath=savepath, epoch=30, learning_rate=.0001, lr_reduce=.995)

models = {
    'RFR': RFR(),
    'DTR': DTR(),
    'pytorch-DNN': tmodel
}

models, predictions = mlutils.train_with_data(outdata, feature_names, models, featimp_models=['RFR', 'DTR'],
                                              figure_dir=savepath, model_type='regressor', evaluate=True)

assert predictions['RFR']['test']['rsquare'] >= .8, 'RFR model test set R-square is less than 0.8'
assert predictions['pytorch-DNN']['test']['rsquare'] >= .7, 'pytorch-DNN model test set R-square is less than .7'
