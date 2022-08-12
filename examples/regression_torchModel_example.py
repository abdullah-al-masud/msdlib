"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""

# torchModel() regression example
import pandas as pd
from sklearn.datasets import fetch_california_housing
import torch

import os
import sys
project_dir = os.getcwd()
sys.path.append(project_dir)
from msdlib import mlutils
from msdlib import msd

# Loading the data and separating data and label
source_data = fetch_california_housing()
feature_names = source_data['feature_names'].copy()
data = pd.DataFrame(source_data['data'], columns=feature_names)
label = pd.Series(source_data['target'], name=source_data['target_names'][0])

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
tmodel = mlutils.torchModel(layers=layers, model_type='regressor', tensorboard_path='runs',
                            savepath='examples/regression_torchModel', epoch=80, learning_rate=.0001, lr_reduce=.995)

# Training Pytorch model
train_set = mlutils.DataSet(torch.tensor(outdata['train']['data'], device='cuda', dtype=torch.float32), 
                            torch.tensor(outdata['train']['label'], device='cuda', dtype=torch.float32).squeeze())
val_set = mlutils.DataSet(torch.tensor(outdata['validation']['data'], device='cuda', dtype=torch.float32), 
                            torch.tensor(outdata['validation']['label'], device='cuda', dtype=torch.float32).squeeze())
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=128)

tmodel.fit(train_loader=train_loader, val_loader=val_loader)

# Evaluating the model's performance
result, all_results = tmodel.evaluate(data_sets=[outdata['train']['data'], outdata['test']['data']],
                                      label_sets=[outdata['train']['label'].ravel(
                                      ), outdata['test']['label'].ravel()],
                                      set_names=['Train', 'Test'], savepath='examples/regression_torchModel')

print('regression result:\n', result)
