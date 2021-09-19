"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""

# torchModel() multi-class classification example
import pandas as pd
from msdlib import msd
from sklearn.datasets import fetch_covtype
import torch
from msdlib import mlutils


# Loading the data and separating data and label
source_data = fetch_covtype()
feature_names = source_data['feature_names'].copy()
data = pd.DataFrame(source_data['data'], columns=feature_names)
label = pd.Series(source_data['target'], name=source_data['target_names'][0])

# print(source_data['DESCR'])
print('data :\n', data.head())
print('labels :\n', label)
print('classes :', label.unique())

# label is 1 based index. So, converting it to 0 based index
label = label - 1

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
                            savepath='multiclass-classification_torchModel', batch_size=512, epoch=80, learning_rate=.0001, lr_reduce=.995)
print(tmodel.model)

# Training Pytorch model
tmodel.fit(outdata['train']['data'], outdata['train']['label'],
           val_data=outdata['validation']['data'], val_label=outdata['validation']['label'])

# Evaluating the model's performance
result, all_results = tmodel.evaluate(data_sets=[outdata['train']['data'], outdata['test']['data']],
                                      label_sets=[
                                          outdata['train']['label'], outdata['test']['label']],
                                      set_names=['Train', 'Test'], savepath='multiclass-classification_torchModel')

print('classification score :\n', result)

# scores for classification
print('test data score :\n', all_results['Test'][0])

# confusion matrix for classification
print('test data confusion matrix :\n', all_results['Test'][1])