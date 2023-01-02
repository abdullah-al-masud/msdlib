"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""


import warnings
try:
    import torch
    from torch import nn
except Exception as e:
    warnings.warn('skipping pytorch importation... If you wish to use pytorch, please install it correctly. If not, please ignore this warning')

import pandas as pd
import matplotlib.pyplot as plt
from ..msd import (
    class_result,
    plot_class_score,
    regression_result,
    plot_regression_score
)
import numpy as np
import os
import joblib

plt.rcParams['figure.facecolor'] = 'white'


try:
    class DataSet(torch.utils.data.Dataset):
        """
        This is a customized Data set object which can build a torch.utils.data.Dataset object given the data and labels. 
        This is only usable when we have complete data and labels (data and label lengths must be equal)

        Inputs:
            :data: ideally should be torch tensor. Contains feature data tor model training.
                Can be python list or python set too but not much appreciated as it will be mostly used for training pytorch model.
            :label: ideally should be numpy ndarray, pandas Series/DataFrame or torch tensor. Contains true labels tor model training.
                    Can be python list or python set too but not much appreciated as it will be mostly used for training pytorch model.
            :dtype: data type of the data and labels. Default is torch.float32. It also depends on model_type parameter for label data.
            :model_type: {'regressor', 'binary-classifier' or 'multi-classifier'}. Default is 'multi-classifier'.
                        It is used to confirm that for multi-class classification, label data type is torch.long.
        """

        def __init__(self, data, label=None, dtype=torch.float32, model_type='regressor'):
            self.data = data
            self.label = label
            if self.label is not None:
                self._check_samples()
            self.datalen = data.shape[0]
            self.dtype = dtype
            self.model_type = model_type

        def __len__(self):
            return self.datalen

        def __getitem__(self, index):
            # data type conversion
            batch_data = self.data[index].to(dtype=self.dtype)

            if self.label is not None:
                if self.model_type.lower() == 'multi-classifier':
                    batch_label = self.label[index].to(dtype=torch.long) 
                else:
                    batch_label = self.label[index].to(dtype=self.dtype)
            else:
                batch_label = None
            return batch_data, batch_label

        def _check_samples(self):
            assert len(self.data) == len(
                self.label), "Data and Label lengths are not same"
except:
    pass


def get_factors(n_layers, base_factor=5, max_factor=10, offset_factor=2):
    """
    This function calculates factors/multipliers to calculate number of units inside define_layers() function

    Inputs:
        :n_layers: number of hidden layers
        :max_factor: multiplier for mid layer (largest layer)
        :base_factor: multiplier for first layer
        :offset_factor: makes assymetric structure in output with factor (base - offset). For symmetric model (size in first and last hidden layer is same), offset will be 0.

    Outputs:
        :factors: multipliers to calculate number of units in each layers based on input shape
    """

    base_factor = max_factor - base_factor
    return [max_factor - abs(x) for x in np.linspace(-base_factor, base_factor + offset_factor, n_layers) if max_factor - abs(x) > 0]


def define_layers(input_units, output_units, unit_factors, dropout_rate=None, model_type='regressor', actual_units=False,
                  apply_bn=False, activation=None, final_activation=None):
    """
    This function takes a common formation/sequence of functions to construct one hidden layer and then replicates this sequence for multiple hidden layers.
    Hidden layer units are decided based on 'unit_factors' paramter.
    The common formation of one hidden layer is this-

        -Linear
        -BatchNorm1d
        -ReLU
        -Dropout

    Dropout ratio is same in all layers

    Finally the output layer is constructed depending on 'output_units' and 'model_type' parameter including final activation function.
    Output activation function is provided  

    Inputs:
        :input_units: int, number of units in input layer / number of features (not first hidden layer)
        :output_units: int, number of units in output layer / number of output nodes / number of classes (not last hidden layer)
        :unit_factors: array of ints or floats, multipliers to calculate number of units in each hidden layer from input_units, or actual number of units for each hidden layer
        :dropout_rate: dropout ratio, must be 0 ~ 1. Default is None (no dropout layer)
        :model_type: {'binary-classifier', 'multi-classifier, 'regressor'}, controls use of softmax/sigmoid at the output layer. Use 'regressor' if you dont intend to use any activation at output. Default is 'regressor'
        :actual_units: bool, whether actual units are placed in unit_factors or not, default is False (not actual units, instead unit_factos is containing ratios)
        :apply_bn: bool, whether to use batch normalization or not, default is False (does not use batch normalization)
        :activation: nn.Module object or None. Pytorch activation layer that will be used as activation function after each hidden layer. Default is None (No activation)
        :final_activation: torch.sigmoid / torch.Softmax(dim=1) / torch.tanh etc. for output layer, default is None. If None, the final activation will be below:
            - modey_type == 'regressor' --> No activation
            - model_type == 'binary-classifier' --> torch.sigmoid
            - model_type == 'multi-clussifier' --> torch.Softmax(dim=1)

    Outputs:
        :layers: list of Deep Learning model layers which can be fed as NNModel layer_funcs input or torchModel layers input to build DNN model
    """

    if actual_units:
        hidden_units = unit_factors.copy()
    else:
        hidden_units = [input_units * factor for factor in unit_factors]
    units = [input_units] + hidden_units + [output_units]
    units = [int(i) for i in units]

    layers = []
    for i in range(len(unit_factors)):
        layers.append(nn.Linear(units[i], units[i + 1]))
        if apply_bn:
            layers.append(nn.BatchNorm1d(units[i + 1]))
        if activation is not None:
            layers.append(activation)
        if dropout_rate is not None:
            layers.append(nn.Dropout(dropout_rate))
    layers.append(nn.Linear(units[-2], units[-1]))
    if final_activation is None:
        if model_type.lower() == 'multi-classifier':
            layers.append(nn.LogSoftmax(dim=1))
        elif model_type.lower() == 'binary-classifier':
            layers.append(torch.sigmoid)
    else:
        layers.append(final_activation)
    return layers


# storing the models and loading them
def store_models(models, folder_path):
    """
    This function stores different types of scikit-learn models in .pickle format and also Pytorch model in .pt format

    Inputs:
        :models: dict, containing only trained model class; {<model name>: <model class>}
                For pytorch models, the key must contain 'pytorch' phrase (Case insensitive). 

                Note: Pytorch model must not be a DataParallel model.\n
                If its a DataParallel model, then take module attribute of your model like this- {'pytorch': <your_model>.module}
        :folder_path: str, the folder path where the models will be stores, if doesnt exist, it will be created
    """

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for modelname in models:
        print('storing models... %s_model...' % modelname, end='')
        if 'pytorch' in modelname.lower():
            torch.save(models[modelname].state_dict(),
                       os.path.join(folder_path, '%s_model.pt' % modelname))
        else:
            with open(os.path.join(folder_path, '%s_model.pickle' % modelname), 'wb') as f:
                joblib.dump(models[modelname], f)
        print('   ...storing completed !!')


def load_models(models, folder_path):
    """
    This function loads different scikit-learn models from .pickle format and Pytorch model from .pt formatted state_dict (only weights)

    Inputs:
        :models: dict, containing model classes or None (for torch model, torch.nn.Module object is necessary as trained model 
                to load the state variables. For other types of models like xgboost etc. None is fine.);
                For pytorch models, the key must contain 'pytorch' phrase (Case insensitive)
                key name must be like this :\n
                    stored model file name: xgboost_model.pickle\n
                    corresponding key for the dict: 'xgboost'\n
                    stored model file name: pytorch-1_model.pickle\n
                    corresponding key for the dict: 'pytorch-1'\n

                for pytorch model, the model must not be a DataParallel model. You can add parallelism after loading the weights
        :folder_path: str, directory path from where the stored models will be loaded

    Outputs:
        :models: dict, containing model classes like {<model_name> : <model_class>}
    """

    for modelname in models:
        print('\rloading models... %s_model...' % modelname, end='')
        if 'pytorch' in modelname.lower():
            models[modelname].load_state_dict(
                torch.load(os.path.join(folder_path, '%s_model.pt' % modelname)))
        else:
            with open(os.path.join(folder_path, '%s_model.pickle' % modelname), 'rb') as f:
                models[modelname] = joblib.load(f)
        print('   ...loading completed !!')
    return models


def train_with_data(outdata, feature_columns, models, featimp_models=[], figure_dir=None, model_type='multi-classifier',
                    evaluate=True, featimp_figsize=(30, 5), figsize=(15, 5), xrot=0):
    """
    This function will be used to train models. We can use both scikit-models and torchModel objects for model training.

    Inputs:
        :outdata: dict, contains dicts with structure like : outdata = {'train': {'data': <numpy-array>, 'label': <numpy-array>, 'index': <numpy-array>}}
                    'train' dict is mandatory. 'validation' dict is mandatory for torchModel objects inside "models" argument.
                    'data', 'label' and 'index' must be of same length.
        :feature_columns: list/numpy array, contains feature names of 'data' inside outdata. seature_columns length must be equal to number of columns in 'data'
        :models: dict, contains scikit-models, xgboost, lightgbm etc. scikit-like models and torchModel objects. 
                    If its a torchModel object, the key must contain 'pytorch' in it.
        :featimp_models: list of strings, contains model names which belong to models dict keys and which can provide feature importances
        :figure_dir: string/None, path to the directory where figures will be saved for feature improtances and evaluation figures.
        :model_type: string, can be either 'regressor', 'multi-classifier' or 'binary-classifier'. It controls the evaluation process.
        :evaluate: boolean, If True, model evaluation will be executed and results of evaluation will be stored inside figure_dir. Default is True.
        :featimp_figsize: tuple of horiaontal and vertical size of the feature importance plot. Default is (30, 5).
        :figsize: tuple of horiaontal and vertical size of the result figure (regression score and classification score both). Default is (15, 5). 
                    Only effective when evaluate is True.
        :xrot: float, rotation angle of the x axis labels in classification score matrix (where class label names are written). Default is 0 (no rotation).
                    Only effective when evaluate is True.

    Outputs:
        :models: dict, contains trained models instances, same as 'models' argument
        :predictions: dict, contains detailed predictions on all sets in outdata argument, evaluation results etc.
    """
    
    if figure_dir is not None:
        os.makedirs(figure_dir, exist_ok=True)
    for modelname in models:
        print('training %s model...    '%modelname, end='', flush=True)
        if 'pytorch' in modelname:
            if figure_dir is not None:
                models[modelname].savepath = figure_dir
            models[modelname] = models[modelname].fit(outdata['train']['data'], np.squeeze(outdata['train']['label']), 
                                                      val_data=outdata['validation']['data'], 
                                                      val_label=np.squeeze(outdata['validation']['label']), evaluate=False)
        else:
            models[modelname] = models[modelname].fit(outdata['train']['data'], np.squeeze(outdata['train']['label']))
        print('  complete !!')

        # feature importance plot
        if figure_dir is not None:
            if modelname in featimp_models:
                fig, ax = plt.subplots(figsize=featimp_figsize)
                feat_imp = pd.Series(models[modelname].feature_importances_, index=feature_columns).sort_values(ascending=False)
                feat_imp.plot(kind='bar', ax=ax, title='Feature importances from %s model'%modelname)
                fig.tight_layout()
                fig.savefig('%s/feature_importances_%s_model.png'%(figure_dir, modelname), bbox_inches='tight')
                plt.close()

    if evaluate:
        predictions = evaluate_with_data(outdata, models, figure_dir, model_type, figsize=figsize, xrot=xrot)
    else:
        predictions = {}

    return models, predictions


def evaluate_with_data(outdata, models, figure_dir=None, model_type='multi-classifier', figsize=(15, 5), xrot=0):
    """
    This function will be used to train models. We can use both scikit-models and torchModel objects for model training.

    Inputs:
        :outdata: dict, contains dicts with structure like : outdata = {'train': {'data': <numpy-array>, 'label': <numpy-array>, 'index': <numpy-array>}}
                    'train' dict is mandatory. 'validation' dict is mandatory for torchModel objects inside "models" argument.
                    'data', 'label' and 'index' must be of same length.
        :models: dict, contains scikit-models, xgboost, lightgbm etc. scikit-like models and torchModel objects. 
                    If its a torchModel object, the key must contain 'pytorch' in it.
        :figure_dir: string/None, path to the directory where figures will be saved for feature improtances and evaluation figures.
        :model_type: string, can be either 'regressor', 'multi-classifier' or 'binary-classifier'. It controls the evaluation process.
        :figsize: tuple of horiaontal and vertical size of the result figure (regression score and classification score both). Default is (15, 5).
        :xrot: float, rotation angle of the x axis labels in classification score matrix (where class label names are written). Default is 0 (no rotation).

    Outputs:
        :predictions: dict, contains detailed predictions on all sets in outdata argument, evaluation results etc.
    """
    
    predictions = {}
    if figure_dir is not None:
        os.makedirs(figure_dir, exist_ok=True)
    for modelname in models:
        # calculating predictions scores
        predictions[modelname] = {}
        for setname in outdata:
            print('predicting and evaluating for %s set from %s model'%(setname, modelname), end='', flush=True)
            predictions[modelname][setname] = {}
            if 'pytorch' in modelname:
                preds = models[modelname].predict(outdata[setname]['data']).detach().cpu().numpy()
                if model_type.lower() == 'binary-classifier':
                    preds = np.squeeze(preds.round())
                elif model_type.lower() == 'multi-classifier':
                    preds = np.squeeze(np.argmax(preds, axis=1))
            else:
                preds = models[modelname].predict(outdata[setname]['data'])
            predictions[modelname][setname]['prediction'] = preds.copy()

            if model_type.lower() in ['binary-classifier', 'multi-classifier']:
                score, confmat = class_result(np.squeeze(outdata[setname]['label']), preds, True)
                predictions[modelname][setname]['score'] = score.copy()
                predictions[modelname][setname]['confusion_matrix'] = confmat.copy()

                if figure_dir is not None:
                    fig_title = 'Classification Score on %s set for %s model'%(setname, modelname)
                    plot_class_score(score, confmat, xrot=xrot, figure_dir=figure_dir, figtitle=fig_title, figsize=figsize)
            elif model_type.lower() == 'regressor':
                rsquare, rmse, corr = regression_result(np.squeeze(outdata[setname]['label']), preds)
                predictions[modelname][setname]['rsquare'] = rsquare
                predictions[modelname][setname]['rmse'] = rmse
                predictions[modelname][setname]['corr'] = corr

                if figure_dir is not None:
                    fig_title = 'Regression Score on %s set for %s model'%(setname, modelname)
                    metrics = {'R-square': rsquare.round(4), 'RMSE': rmse.round(4), 'Corr. Coefficient': corr.round(4)}
                    plot_regression_score(np.squeeze(outdata[setname]['label']), preds, figure_dir=figure_dir,
                                              figtitle=fig_title, figsize=figsize, metrics=metrics)

            print('  complete !!')
    
    return predictions
