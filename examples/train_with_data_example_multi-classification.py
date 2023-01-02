"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""

# torchModel() multi-class classification example
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.tree import DecisionTreeClassifier as DTC
import pandas as pd
from sklearn.datasets import load_digits
import torch

import os
import sys
project_dir = os.getcwd()
sys.path.append(project_dir)
from msdlib import mlutils
from msdlib import msd


savepath = 'examples/train_with_data_multi-classification'
# Loading the data and separating data and label
source_data = load_digits()
feature_names = source_data['feature_names'].copy()
data = pd.DataFrame(source_data['data'], columns=feature_names)
label2index = {name: i for i, name in enumerate(source_data['target_names'])}
label = pd.Series(source_data['target']).replace(label2index)
# print(source_data['DESCR'])
print('data :\n', data.head())
print('labels :\n', label)
print('classes :', label.unique())

# Standardizing numerical data
data = msd.standardize(data)

# Splitting data set into train, validation and test
splitter = msd.SplitDataset(data, label, test_ratio=.1)
outdata = splitter.random_split(val_ratio=.1)

print("outdata.keys() :", outdata.keys())
print("outdata['train'].keys() :", outdata['train'].keys())
print("outdata['validation'].keys() :", outdata['validation'].keys())
print("outdata['test'].keys() :", outdata['test'].keys())
print("train > data, labels and index shapes :",
      outdata['train']['data'].shape, outdata['train']['label'].shape, outdata['train']['index'].shape)
print("validation > data, labels and index shapes :",
      outdata['validation']['data'].shape, outdata['validation']['label'].shape, outdata['validation']['index'].shape)
print("test > data, labels and index shapes :",
      outdata['test']['data'].shape, outdata['test']['label'].shape, outdata['test']['index'].shape)

# defining layers inside a list
layers = mlutils.define_layers(data.shape[1], label.unique().shape[0], [100, 100, 100, 100, 100, 100], dropout_rate=.2,
                               actual_units=True, activation=torch.nn.ReLU(), model_type='regressor')

tmodel = mlutils.torchModel(layers=layers, model_type='multi-classifier',
                            savepath=savepath, batch_size=64, epoch=100, learning_rate=.0001, lr_reduce=.995)

models = {
    'RandomForest': RFC(),
    'DecisionTree': DTC(),
    'pytorch-DNN': tmodel
}

models, predictions = mlutils.train_with_data(outdata, feature_names, models, featimp_models=['RandomForest', 'DecisionTree'],
                                              figure_dir=savepath, model_type='multi-classifier', evaluate=True, figsize=(35, 5))

assert predictions['RandomForest']['test']['score']['average'].loc['f1_score'] >= .9, 'RandomForest model test set f1-score is less than .9'
assert predictions['pytorch-DNN']['test']['score']['average'].loc['f1_score'] >= .9, 'pytorch-DNN model test set f1-score is less than .9'
