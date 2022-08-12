"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""

# torchModel() binary classification example
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.tree import DecisionTreeClassifier as DTC
import pandas as pd
from sklearn.datasets import load_breast_cancer
import torch

import os
import sys
project_dir = os.getcwd()
sys.path.append(project_dir)
from msdlib import msd
from msdlib import mlutils


savepath = 'examples/train_with_data_binary-classification'

# Loading the data and separating data and label
source_data = load_breast_cancer()
feature_names = source_data['feature_names'].copy()
data = pd.DataFrame(source_data['data'], columns=feature_names)
label = pd.Series(source_data['target'], name=source_data['target_names'][1])

# print(source_data['DESCR'])
print('data:', data.head())
print('label', label)
print('classes:', label.unique())


# Standardizing numerical data
data = msd.standardize(data)

# Splitting data set into train, validation and test
splitter = msd.SplitDataset(data, label, test_ratio=.1)
outdata = splitter.random_split(val_ratio=.1)
print('outdata keys:', outdata.keys())
print('outdata["train"].keys() :', outdata['train'].keys())
print("outdata['validation'].keys() :", outdata['validation'].keys())
print("outdata['test'].keys() :", outdata['test'].keys())
print('train > data, label and index shapes',
      outdata['train']['data'].shape, outdata['train']['label'].shape, outdata['train']['index'].shape)
print('validation > data, label and index shapes',
      outdata['validation']['data'].shape, outdata['validation']['label'].shape, outdata['validation']['index'].shape)
print('test > data, label and index shapes',
      outdata['test']['data'].shape, outdata['test']['label'].shape, outdata['test']['index'].shape)


# defining layers inside a list
layers = mlutils.define_layers(data.shape[1], 1, [100, 100, 100, 100, 100, 100], dropout_rate=.2, model_type='binary-classifier',
                               actual_units=True, activation=torch.nn.ReLU(), final_activation=torch.nn.Sigmoid())

# building model
tmodel = mlutils.torchModel(layers=layers, model_type='binary-classifier',
                            savepath=savepath, batch_size=32, epoch=60, learning_rate=.0001, lr_reduce=.995)

models = {
    'RFC': RFC(),
    'DTC': DTC(),
    'pytorch-DNN': tmodel
}

models, predictions = mlutils.train_with_data(outdata, feature_names, models, featimp_models=['RFC', 'DTC'],
                                              figure_dir=savepath, model_type='binary-classifier', evaluate=True)
