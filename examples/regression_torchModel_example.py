"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""

# torchModel() regression example
import pandas as pd
from msdlib import msd
from sklearn.datasets import fetch_california_housing
import torch
from msdlib import mlutils


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
tmodel = mlutils.torchModel(layers=layers, model_type='regressor',
                            savepath='regression_torchModel', epoch=80, learning_rate=.0001, lr_reduce=.995)

# Training Pytorch model
tmodel.fit(outdata['train']['data'], outdata['train']['label'],
           val_data=outdata['validation']['data'], val_label=outdata['validation']['label'])

# Evaluating the model's performance
result, all_results = tmodel.evaluate(data_sets=[outdata['train']['data'], outdata['test']['data']],
                                      label_sets=[outdata['train']['label'].ravel(
                                      ), outdata['test']['label'].ravel()],
                                      set_names=['Train', 'Test'], savepath='regression_torchModel')

print('regression result:\n', result)
