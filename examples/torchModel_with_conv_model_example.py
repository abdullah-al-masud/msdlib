"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""

# torchModel() multi-class classification with convolutional NN example
import numpy as np
import pandas as pd
import os
import cv2
import torch
from torch import nn
from tqdm import tqdm

import os
import sys
project_dir = os.getcwd()
sys.path.append(project_dir)
from msdlib import mlutils
from msdlib import msd


# Loading the data and separating data and label
def select_color(_img, color_index, channel_index=2):
    img = _img.copy()
    mask = np.argmax(img, axis=channel_index) == color_index
    img[:, :, color_index][mask] = 255
    return img


def process_image(image, size=80):
    filt_image = select_color(image, 1)
    return cv2.resize(filt_image, (size, size), interpolation=cv2.INTER_AREA)


# data set is downloaded from here- https://www.kaggle.com/c/plant-seedlings-classification/data
path = 'E:/Data sets/plant-seedlings-classification/train'
fixed_size = 80

# loading data
images = []
labels = []
shapes = []
for cls in tqdm(os.listdir(path)):
    clspath = os.path.join(path, cls)
    if os.path.isdir(clspath):
        for im in os.listdir(clspath):
            images.append(process_image(cv2.imread(
                os.path.join(clspath, im)), fixed_size))
            labels.append(cls)
            _shape = images[-1].shape
            if _shape not in shapes:
                shapes.append(_shape)
images = np.stack(images, axis=0)
labels = np.array(labels)
print('images.shape, labels.shape :', images.shape, labels.shape)
print("classes :", np.unique(labels))
print('labels :\n', labels)

# repalcing label names by label index
label2index = {name: i for i, name in enumerate(np.sort(np.unique(labels)))}
index2label = {i: name for name, i in label2index.items()}
print('label to index conversion :\n', label2index)

splitter = msd.SplitDataset(images, labels, test_ratio=.1, same_ratio=True)
outdata = splitter.random_split(val_ratio=.1)
del images, labels

print("outdata.keys() :", outdata.keys())
# Standardizing numerical data
for s in outdata.keys():
    if s in ['train', 'validation']:
        outdata[s]['data'] = np.concatenate(
            (outdata[s]['data'], np.flip(outdata[s]['data'], axis=1)), axis=0)
        outdata[s]['data'] = np.concatenate(
            (outdata[s]['data'], np.flip(outdata[s]['data'], axis=2)), axis=0)
        outdata[s]['label'] = torch.tensor(
            pd.Series(outdata[s]['label'][:, 0]).replace(label2index).values).repeat(4)
    if s == 'test':
        outdata[s]['label'] = torch.tensor(
            pd.Series(outdata[s]['label'][:, 0]).replace(label2index).values)
    outdata[s]['data'] = torch.tensor(outdata[s]['data']).transpose(1, 3) / 255


# Splitting data set into train, validation and test
print("outdata['train'].keys() :", outdata['train'].keys())
print("outdata['validation'].keys() :", outdata['validation'].keys())
print("outdata['test'].keys() :", outdata['test'].keys())
print("train > data, labels and index shapes :",
      outdata['train']['data'].shape, outdata['train']['label'].shape, outdata['train']['index'].shape)
print("validation > data, labels and index shapes :",
      outdata['validation']['data'].shape, outdata['validation']['label'].shape, outdata['validation']['index'].shape)
print("test > data, labels and index shapes :",
      outdata['test']['data'].shape, outdata['test']['label'].shape, outdata['test']['index'].shape)


# Creating Pytorch model
ch1 = 64
k1 = 5
pool1 = 2
ch2 = 32
k2 = 3
pool2 = 2
ch3 = 16
k3 = 3
pool3 = 2
ln0 = (((outdata['train']['data'].shape[2] - k1 + 1) // pool1 - k2 + 1) // pool2 - k3 + 1) // pool3 * \
    (((outdata['train']['data'].shape[3] - k1 + 1) //
     pool1 - k2 + 1) // pool2 - k3 + 1) // pool3 * ch3
ln1 = 256
drop = .5
ln2 = 128
nout = len(label2index)
layers = [
    nn.Conv2d(outdata['train']['data'].shape[1], ch1, k1),
    nn.MaxPool2d(pool1),
    nn.BatchNorm2d(ch1),
    nn.ReLU(),
    nn.Conv2d(ch1, ch2, k2),
    nn.MaxPool2d(pool2),
    nn.BatchNorm2d(ch2),
    nn.ReLU(),
    nn.Conv2d(ch2, ch3, k3),
    nn.MaxPool2d(pool3),
    nn.BatchNorm2d(ch3),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(ln0, ln1),
    nn.ReLU(),
    nn.Dropout(drop),
    nn.Linear(ln1, ln2),
    nn.ReLU(),
    nn.Linear(ln2, nout),
    # nn.Softmax(dim=1)
]

tmodel = mlutils.torchModel(layers=layers, model_type='multi-classifier',
                            savepath='examples/multiclass-classification_torchModel_conv', batch_size=32, epoch=30, 
                            learning_rate=.0001, lr_reduce=.995, tensorboard_path='runs')
print(tmodel.model)

# Training Pytorch model
tmodel.fit(outdata['train']['data'], outdata['train']['label'],
           val_data=outdata['validation']['data'], val_label=outdata['validation']['label'])

# Evaluating the model's performance
result, all_results = tmodel.evaluate(data_sets=[outdata['train']['data'], outdata['test']['data']],
                                      label_sets=[outdata['train']['label'], outdata['test']['label']],
                                      set_names=['Train', 'Test'], 
                                      savepath='examples/multiclass-classification_torchModel_conv', figsize=(30, 5))
print('classification scores :\n', result)

# scores for classification
print("test data scores :\n", all_results['Test'][0])

# confusion matrix for classification
print("test data confusion matrix :\n", all_results['Test'][1])

assert all_results['Test'][0]['average'].loc['f1_score'] >= .90, 'test set f1-score is less than .90'
