"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""


import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import matplotlib.pyplot as plt
from ..msd import (
    get_time_estimation,
    rsquare_rmse,
    class_result,
    plot_heatmap
)
import numpy as np
import os
import time
from ..mlutils.utils import DataSet, store_models

plt.rcParams['figure.facecolor'] = 'white'


class NNmodel(nn.Module):
    """
    This class constructs a deep neural network model upon providing the layers we intend to build the model with.

    Inputs:
        :layer_funcs: list, contains sequential layer classes (nn.Module). 

            For example-\n
            [nn.Linear(50), nn.ReLU(), nn.Linear(3), nn.Softmax(dim=-1)]

        :seed_value: float/int, random seed for reproducibility, default is 1216
    """

    def __init__(self, layer_funcs, seed_value=1216):

        super(NNmodel, self).__init__()
        # reproducibility parameters
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.model = nn.Sequential(*layer_funcs)

    def forward(self, x):
        """
        pytorch forward function for forward propagation
        """

        x = self.model(x)
        return x


# The class can be used for auto-encoder architecture
class AutoEncoderModel(nn.Module):
    """
    This class creates an auto-encoder model.

    Inputs:
        :enc_layers: python list, containing the encoder layers (torch.nn.Module class objects) sequentially
        :dec_layers: python list, containing the decoder layers (torch.nn.Module class objects) sequentially
        :seed_value: float/int, random seed for reproducibility
    """

    def __init__(self, enc_layers, dec_layers, seed_value=1216):

        super(AutoEncoderModel, self).__init__()
        # reproducibility parameters
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.encode_layers = nn.Sequential(*enc_layers)
        self.decode_layers = nn.Sequential(*dec_layers)
        self.enc_len = len(self.encode_layers)
        self.dec_len = len(self.decode_layers)

    def encode(self, x):
        """
        Encoder part in Autoencoder model

        Inputs:
            :x: input tensor for encoder part

        Outputs:
            encoder output
        """

        for layer in self.encode_layers:
            x = layer(x)
        return x

    def decode(self, x):
        """
        Decoder part in Autoencoder model

        Inputs:
            :x: input tensor for decoder part

        Outputs:
            decoder output
        """

        for layer in self.decode_layers:
            x = layer(x)
        return x

    def forward(self, x):
        """
        pytorch forward function for forward propagation, applies encoder and then decoder sequentially on the input data

        Inputs:
            :x: input tensor for autoencoder model 

        Outputs:
            :x: final output of the whole auto-encoder model
        """

        x = self.encode(x)
        x = self.decode(x)
        return x


# torchModel is a scikit like wrapper for pytorch which enables us to use the model for
# training, predicting and evaluating performance using simple fit, predict and evaluate methods
class torchModel():
    """
    This class controls the deep neural network model training, inference, prediction and evaluation.
    It can provide the training loss curves and evaluation results very nicely. It is capable of using multiple GPUs.

    Note: For classification problems (both binary and multi-class), there is no need to convert labels into one hot encoded format. 
    In stead, the class label values should be indices like 0, 1, 2...

    Inputs:
        :layers: a list of torch.nn.Module objects indicating layers/activation functions. The list should contain all elements sequentially. 
                 Default is []. Each element of this list must be callable function object. 
                 For Sigmoid function, we can use torch.nn.Sigmoid() or torch.sigmoid (use the brackets this way).
        :loss_func: loss function for the ML model. default is torch.nn.MSELoss [without brackets ()]. It can also be a custom loss function, but should be equivalent to the default
        :optimizer: optimizer for the ML model. default is torch.optim.Adam
        :learning_rate: learning rate of the training steps, default is .0001
        :epoch: number of epoch for training, default is 2
        :batch_size: mini-batch size for trianing, default is 32
        :lr_reduce: learning rate reduction base for lambda reduction scheduler from pytorch (follows torch.optim.lr_scheduler.LambdaLR). 
                    Must be 0 ~ 1. Default is 1 (No reduction of learning rate during training)
        :loss_reduction: loss reduction parameter for loss calculation, default is 'mean'
        :model_type: type of the model depending on the objective, should be any of {'regressor', 'binary-classifier', 'multi-classifier'}, default is 'regressor'. 
                    - 'binary-classifier' indicates binary classifier with 1 output unit. The output values must be 0 ~ 1 (sigmoid like activations should be used at model output). 
                    - 'multi-classifier' indicates classifier with more than one output unit
        :use_gpu: bool, whether to use gpu or not, default is True.
        :gpu_devices: list, a list of cuda ids starting from 0. Default is None which will try to use all available gpus. 
                    If you have 4 gpus in your machine, and want to use first 3 of them, the list should be [0, 1, 2]
                    It can also be used to select a particular GPU device. For example to use GPU device 1, make it [1]
        :model_name: str, name of the model, default is 'pytorch'
        :dtype: dtype of processing inside the model, default is torch.float32
        :plot_loss: bool, whether to plot loss curves after training or not, default is True
        :quant_perc: float, quantile value to limit the loss values for loss curves, default is .98
        :plot_evaluation: bool, whether to plot evaluation tables/figures or not, default is True
                        - For model_type={binary-classifier, multi-classifier}, it will be score matrix and confusion matrix plot
                        - For model_type=regressor, it will be a true vs prediction scatter plot
        :loss_roll_preiod: int, rolling/moving average period for loss curve
        :model: torch.nn.Module class (ML model class), so that we are able to write the model ourselves and use fit, predict etc methods from here.  
        :savepath: str, path to store the learning curve and evaluation results
        :shuffle: bool, whether the training dataset should be shuffled during training or not. Default is True (data set will be shuffles). 
        :tensorboard_path: str or None, tensorboard will show the progress of training using loss values and training metrics.
                            - If model_type is 'regressor', metrics will be rmse, rsquare
                            - If model_type is 'binary-classifier' or 'multi-classifier', metrics will be 'accuracy'
        :tb_metrics: None or list of functions. Each function will take two inputs, first is true labels and second is predicted values.
        			If None, the metrics will be selected based on model type according to the description of parameter 'tensorboard_path'.
        :interval: int or None, indicates number of epoch after which the model weights will be stored each time during training.
                    Default is None means model will not be stored during training. Note: "model_name" parameter must include 'pytorch' phrase in it.
    """

    def __init__(self, layers=[], loss_func=None, optimizer=None, learning_rate=.0001, epoch=2, batch_size=32, lr_reduce=1,
                 loss_reduction='mean', model_type='regressor', use_gpu=True, gpu_devices=None, model_name='pytorch', dtype=torch.float32,
                 plot_loss=True, quant_perc=.98, plot_evaluation=True, loss_roll_period=1, model=None, savepath=None, shuffle=True, lr_scheduler=None,
                 tensorboard_path=None, tb_metrics=None, interval=None, **kwargs):

        # defining model architecture
        if model is None and len(layers) == 0:
            raise ValueError('Must provide either "layers" or "model" argument')
        self.model = NNmodel(layers) if model is None else model

        # defining training formation parameters
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.lr_reduce = lr_reduce
        self.loss_reduction = loss_reduction
        self.model_type = model_type
        self.use_gpu = use_gpu
        self.model_name = model_name
        self.dtype = dtype
        self.savepath = savepath
        self.gpu_devices = gpu_devices
        self.shuffle = shuffle
        self.lr_scheduler = lr_scheduler

        # evaluation parameters
        self.plot_loss = plot_loss
        self.quant_perc = quant_perc
        self.plot_evaluation = plot_evaluation
        self.loss_roll_period = loss_roll_period
        self.tb = None
        if tensorboard_path is not None:
            self.tb = SummaryWriter(log_dir=tensorboard_path)
        self.tb_metrics = tb_metrics
        self.interval = interval

        # setting up gpu usage
        self.gpu_string = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
        if self.gpu_string == 'cuda':
            if isinstance(self.gpu_devices, list):
                self.device = torch.device('cuda:%d' % self.gpu_devices[0])
            else:
                self.device = torch.device(self.gpu_string)
        else:
            self.device = torch.device('cpu')
        self.model = self.model.to(device=self.device, dtype=self.dtype)

        # setting up loss and optimizer
        if loss_func is None:
            if model_type.lower() == 'regressor':
                self.loss_func = nn.MSELoss
            elif model_type.lower() == 'binary-classifier':
                self.loss_func = nn.BCELoss
            elif model_type.lower() == 'multi-classifier':
                self.loss_func = nn.CrossEntropyLoss
            else:
                raise ValueError('model_type %s is not valid !' % model_type)
        else:
            self.loss_func = loss_func
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam
        self.loss_func = self.loss_func(reduction=self.loss_reduction).to(
            device=self.device, dtype=self.dtype)
        self.optimizer = self.optimizer(
            self.model.parameters(), lr=self.learning_rate)

        # learning rate scheduler
        if self.lr_scheduler is None:
            def lr_lambda(ep): return self.lr_reduce ** ep
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda)

        self.parallel = False  # whether DataParallel will be used or not

    def set_parallel(self,):
        """
        This method sets multiple GPUs to use for training depending on the machine's cuda availability and use_gpu parameter.
        """

        if self.gpu_string == 'cuda':
            print('Cuda is available and will be used for training')
            if self.gpu_devices is None:
                self.gpu_devices = [i for i in range(torch.cuda.device_count())]
            if isinstance(self.gpu_devices, list):
                if len(self.gpu_devices) > 1:
                    self.model = nn.DataParallel(
                        self.model, device_ids=self.gpu_devices)
                    self.parallel = True
                print('Training will be done on these cuda devices:', self.gpu_devices)
        else:
            print('Cuda/GPU isnt detected, training will be done in CPU')

    def relieve_parallel(self):
        """
        This function reverts the DataParallel model to simple torch.nn.Module model
        """

        if self.parallel:
            self.model = self.model.module

    def fit(self, data=None, label=None, val_data=None, val_label=None, validation_ratio=.15, evaluate=True, figsize=(18, 4), train_loader=None, val_loader=None):
        """
        scikit like wrapper for training DNN pytorch model

        Inputs:

            :data: input train data, must be torch tensor, numpy ndarray or pandas DataFrame/Series. Default value in None
            :label: supervised labels for data, must be torch tensor, numpy ndarray or pandas DataFrame/Series. Default value in None
            :val_data: validation data, must be torch tensor, numpy ndarray or pandas DataFrame/Series, default is None
            :val_label: supervised labels for val_data, must be torch tensor, numpy ndarray or pandas DataFrame/Series, default is None
            :validation_ratio: ratio of 'data' that will be used for validation during training. It will be used only when val_data or val_label or both are None. Default is 0.15
            :evaluate: bool, whether to evaluate model performance after training ends or not. evaluate performance if set True. Default is True.
            :figsize: tuple of (width, height) of the figure, size of the figure for loss curve plot and evaluation plots. Default is (18, 4)
            :train_loader: torch.utils.data.DataLaoder instance for handling training data set. Default is None
            :val_loader: torch.utils.data.DataLaoder instance for handling validation data set. Default is None

        Outputs:
            :self: torchModel object, returns the self object variable just like scikit-learn model objects
        """

        train_loader, val_loader = self.prepare_data_loader(
            data, label, val_data, val_label, validation_ratio, train_loader, val_loader)

        # running through epoch
        loss_curves = [[], []]
        val_loss = torch.tensor(np.nan)
        t1 = time.time()
        self.set_parallel()
        total_batch = len(train_loader)
        for ep in range(self.epoch):
            tr_mean_loss = []
            self.model.train()
            for i, (batch_data, batch_label) in enumerate(train_loader):
                # loss calculation
                self.model.zero_grad()
                label_hat = self.model(batch_data.to(device=self.device, dtype=self.dtype))
                tr_loss = self.loss_func(label_hat.squeeze(), batch_label.to(device=self.device).squeeze())

                # back-propagation
                tr_loss.backward()
                # model parameter update
                self.optimizer.step()

                # stacking and printing losses
                tr_mean_loss.append(tr_loss.item())
                time_string = get_time_estimation(
                    time_st=t1, current_ep=ep, current_batch=i, total_ep=self.epoch, total_batch=total_batch)
                print('\repoch : %04d/%04d, batch : %03d, train_loss : %.4f, validation_loss : %.4f,  %s'
                      % (ep + 1, self.epoch, i + 1, tr_loss.item(), val_loss.item(), time_string)+' '*20, end='', flush=True)

            # loss scheduler step
            self.scheduler.step()
            # storing losses
            tr_mean_loss = np.mean(tr_mean_loss)
            loss_curves[0].append(tr_mean_loss)

            if len(val_loader) > 0:
                # run evaluation to get validation score
                out, _val_label = self.predict(val_loader, return_label=True)
                val_loss = self.loss_func(out.squeeze(), _val_label.squeeze())
                # storing losses
                loss_curves[1].append(val_loss.item())
            else:
                loss_curves[1].append(np.nan)

			# tensorboard parameter collection
            self.add_tb_params(ep, tr_mean_loss, val_loss, _val_label.squeeze(), out, batch_data)

            # storing model weights after each interval
            if self.interval is not None:
                if (ep + 1) % self.interval == 0:
                    if self.parallel:
                        model_dict = {"%s_epoch-%d"%(self.model_name, ep+1): self.model.module}
                    else:
                        model_dict = {"%s_epoch-%d"%(self.model_name, ep+1): self.model}
                    store_models(model_dict, self.savepath)
	
        print('...training complete !!')
        losses = pd.DataFrame(loss_curves, index=['train_loss', 'validation_loss'], columns=np.arange(
            1, self.epoch + 1)).T.rolling(self.loss_roll_period).mean()

        # plotting loss curve
        if self.plot_loss and self.epoch > 1:
            ylim_upper = losses.quantile(self.quant_perc).max()
            ylim_lower = losses.min().min()
            fig, ax = plt.subplots(figsize=(25, 4))
            losses.plot(ax=ax, color=['darkcyan', 'crimson'])
            ax.set_ylim(ylim_lower, ylim_upper)
            fig.suptitle('Learning curves', y=1,
                         fontsize=15, fontweight='bold')
            fig.tight_layout()
            if self.savepath is not None:
                os.makedirs(self.savepath, exist_ok=True)
                fig.savefig('%s/Learning_curves.png' %
                            (self.savepath), bbox_inches='tight')
                plt.close()
            else:
                plt.show()

        self.relieve_parallel()

        if self.interval is not None:
            model_dict = {self.model_name: self.model}
            store_models(model_dict, self.savepath)

        # model training evaluation
        if evaluate:
            self.evaluate([train_loader.dataset.data, val_loader.dataset.data], 
                          [train_loader.dataset.label, val_loader.dataset.label], 
                          set_names=['Train_set (from training data)', 'Validation_set (from training data)'], 
                          figsize=figsize, savepath=self.savepath)

        return self

    def predict(self, data, return_label=False):
        """
        A wrapper function that generates prediction from pytorch model

        Inputs:
            :data: input data to predict on, can be a torch tensor, numpy ndarray, pandas DataFrame or even torch DataLoader object
            :return_label: bool, whether to return label data if 'data' is a pytorch DataLoader object. Default is False

        Outputs:
            :preds: torch Tensor, predicted values against the inserted data
            :labels: torch Tensor, only found if 'data' is pytorch DataLoader and return_label is True.
        """

        # evaluation mode set up
        self.model.eval()

        if isinstance(data, torch.utils.data.dataloader.DataLoader):
            with torch.no_grad():
                preds = []
                labels = []
                for i, (batch, label) in enumerate(data):
                    pred = self.model(batch.to(device=self.device, dtype=self.dtype))
                    preds.append(pred.detach())
                    if return_label:
                        labels.append(label)
                preds = torch.cat(preds)
                if return_label:
                    labels = torch.cat(labels).to(device=self.device)
                    if self.model_type.lower() == 'multi-classifier':
                        labels = labels.to(dtype=torch.long)
                    else:
                        labels = labels.to(dtype=self.dtype)
                    return preds, labels
                else:
                    return preds
        
        else:
            # checking data type
            if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
                data = data.values
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)

            # estimating number of mini-batch
            n_batch = data.shape[0] // self.batch_size + \
                int(bool(data.shape[0] % self.batch_size))

            with torch.no_grad():
                # generates prediction
                preds = []
                for i in range(n_batch):
                    if i != n_batch - 1:
                        pred = self.model(
                            data[i * self.batch_size: (i + 1) * self.batch_size].to(device=self.device, dtype=self.dtype))
                    else:
                        pred = self.model(
                            data[i * self.batch_size:].to(device=self.device, dtype=self.dtype))
                    preds.append(pred.detach())
                preds = torch.cat(preds)
            
            return preds

    def evaluate(self, data_sets, label_sets, set_names=[], figsize=(18, 4), savepath=None):
        """
        This is a customized function to evaluate model performance in regression and classification type tasks

        Inputs:
            :data_sets: list of data, data must be numpy ndarray, torch tensor or Pandas DataFrame/Series
            :label_sets: list of labels corresponding to each data, label must be numpy ndarray, torch tensor or Pandas DataFrame/Series
            :set_names: names of the data sets, default is []
            :figsize: figure size for the evaluation plots, default is (18, 4)
            :savepath: path where the evaluation tables/figures will be stored, default is None

        Outputs:
            :summary_result: pandas DataFrame, result summary accumulated in a variable
            :all_results: dict, complete results for all datasets
        """

        results, all_results = None, None
        # plotting true vs prediction curve (regression) or confusion matrix (classification)
        if self.plot_evaluation and self.model_type.lower() in ['regressor', 'binary-classifier', 'multi-classifier'] and len(data_sets) > 0:
            set_names = set_names if len(set_names) > 0 else [
                'data-%d' % (i+1) for i in range(len(data_sets))]
            all_results = {}
            results = []
            for i, (preddata, predlabel) in enumerate(zip(data_sets, label_sets)):
                test_pred = self.predict(
                    preddata).detach().cpu().squeeze().numpy()
                if isinstance(predlabel, torch.Tensor):
                    label = predlabel.detach().cpu().squeeze().numpy()
                elif isinstance(predlabel, pd.DataFrame) or isinstance(predlabel, pd.Series):
                    label = predlabel.values.copy()
                else:
                    label = predlabel.copy()
                if self.model_type.lower() == 'regressor':
                    true_pred = pd.DataFrame([label, test_pred], index=[
                                             'true_label', 'prediction']).T
                    corr_val = true_pred.corr().iloc[0, 1]
                    rsquare, rmse = rsquare_rmse(
                        true_pred['true_label'].values, true_pred['prediction'].values)
                    fig, ax = plt.subplots(figsize=figsize)
                    ax.scatter(
                        true_pred['true_label'], true_pred['prediction'], color='darkcyan', s=8)
                    _min = np.min([true_pred['true_label'].min(),
                                  true_pred['prediction'].min()])
                    _max = np.max([true_pred['true_label'].max(),
                                  true_pred['prediction'].max()])
                    ax.plot([_min, _max], [_min, _max], color='k', lw=2)
                    ax.set_xlabel('true-label')
                    ax.set_ylabel('prediction')
                    title = 'True-Label VS Prediction Scatter plot for %s from %s\nRSquare : %.3f,  RMSE : %.3f,  Correlation : %.3f' % (
                        set_names[i], self.model_name, rsquare, rmse, corr_val)
                    ax.set_title(title, fontweight='bold')
                    all_results[set_names[i]] = [rsquare, rmse]
                    results.append(pd.Series([rsquare, rmse], index=[
                                   'r_square', 'rmse'], name='%s_%s' % (self.model_name, set_names[i])))
                elif self.model_type.lower() in ['multi-classifier', 'binary-classifier']:
                    if self.model_type.lower() == 'multi-classifier':
                        test_pred = np.argmax(test_pred, axis=1)
                    else:
                        test_pred = np.round(test_pred).astype(int)
                    result, confus = class_result(
                        label, test_pred, out_confus=True)
                    fig, ax = plt.subplots(figsize=figsize, ncols=2)
                    ax[0] = plot_heatmap(result, annotate=True, fmt='.3f', xrot=0,
                                             vmax=1, axobj=ax[0], cmap='summer', fig_title='Score Matrix')
                    ax[1] = plot_heatmap(
                        confus, annotate=True, fmt='d', xrot=0, axobj=ax[1], cmap='Blues', fig_title='Confusion Matrix')
                    title = 'Classification result for %s from %s' % (
                        set_names[i], self.model_name)
                    fig.suptitle(title, fontsize=15, fontweight='bold')
                    all_results[set_names[i]] = [result, confus]
                    results.append(pd.Series(result.mean().drop('average').to_list() + [result['average'].loc['accuracy']], index=result.drop('average', axis=1).columns.to_list() + ['average'],
                                             name='%s_%s' % (self.model_name, set_names[i])))
                fig.tight_layout()
                if savepath is not None:
                    os.makedirs(savepath, exist_ok=True)
                    fig.savefig(
                        '%s/%s.png' % (savepath, title.split('\n')[0].replace(' ', '_')), bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()
            results = pd.concat(results, axis=1, sort=False).T

        return results, all_results

    def prepare_data_loader(self, data, label, val_data, val_label, validation_ratio, train_loader, val_loader):
        """
        This function prepares the data format and creates training and validation data loader from training and validation data. 
        If no validation data is provided, it uses validation_ratio to split training data into train and validation and creates separate data loader for each set.
        If direct train and validation data loaders are provided, then no additional processing is used.

        Inputs:
            :train_data: input training data, must be torch tensor, numpy ndarray or pandas DataFrame/Series. Default value in None
            :train_label: supervised labels for train_data, must be torch tensor, numpy ndarray or pandas DataFrame/Series. Default value in None
            :val_data: validation data, must be torch tensor, numpy ndarray or pandas DataFrame/Series, default is None.
                        Note: If val_data is not provided, validation data will be taken from training data set using validation_ratio.
            :val_label: supervised labels for val_data, must be torch tensor, numpy ndarray or pandas DataFrame/Series, default is None
            :validation_ratio: ratio of 'data' that will be used for validation during training. It will be used only when val_data or val_label or both are None. Default is 0.15
            :train_loader: torch.utils.data.DataLaoder instance for handling training data set or None. Default is None.
                           Note: If train_loader is provided, its expected that val_loader will also be provided.
            :val_loader: torch.utils.data.DataLaoder instance for handling validation data set or None. Default is None

        Outputs:
            :train_loader: torch.utils.data.DataLaoder instance for handling training data set.
            :val_loader: torch.utils.data.DataLaoder instance for handling validation data set.
        """

        if train_loader is None:
            # allowing pandas dataframe or series input
            if isinstance(data, pd.DataFrame):
                data = data.values
            elif isinstance(data, pd.Series):
                data = data.to_frame().values
            if isinstance(label, pd.DataFrame):
                label = label.values
            elif isinstance(label, pd.Series):
                label = label.values

            if not (isinstance(data, np.ndarray) or isinstance(data, torch.Tensor)):
                raise ValueError(
                    'provided data has invalid format (valid formats are: torch tensor, numpy array or pandas dataframe/Series)')
            if not (isinstance(label, np.ndarray) or isinstance(label, torch.Tensor)):
                raise ValueError(
                    'provided label has invalid format (valid formats are: torch tensor, numpy array or pandas dataframe/Series)')

            data = data.squeeze()
            label = label.squeeze()

        if val_loader is None:
            if isinstance(val_data, pd.DataFrame):
                val_data = val_data.values
            elif isinstance(val_data, pd.Series):
                val_data = val_data.to_frame().values
            if isinstance(val_label, pd.DataFrame):
                val_label = val_label.values
            elif isinstance(val_label, pd.Series):
                val_label = val_label.values

            if val_data is None or val_label is None:
                if data is not None and label is not None:
                    # splitting data set
                    train_ratio = 1 - validation_ratio
                    idx = np.arange(label.shape[0])
                    np.random.shuffle(idx)
                    train_idx = idx[:int(train_ratio * label.shape[0])]
                    val_idx = idx[int(train_ratio * label.shape[0]):]

                    train_data = data[train_idx]
                    train_label = label[train_idx]
                    val_data = data[val_idx]
                    val_label = label[val_idx]
            else:
                train_data = data
                train_label = label

            if not (isinstance(val_data, np.ndarray) or isinstance(val_data, torch.Tensor)):
                raise ValueError(
                    'provided validation data has invalid format (valid formats are: torch tensor, numpy array or pandas dataframe/Series)')
            if not (isinstance(val_label, np.ndarray) or isinstance(val_label, torch.Tensor)):
                raise ValueError(
                    'provided validation label has invalid format (valid formats are: torch tensor, numpy array or pandas dataframe/Series)')
            val_data = val_data.squeeze()
            val_label = val_label.squeeze()
        else:
            train_data = data
            train_label = label

        if train_loader is None:
            # handling data sets and labels
            if not isinstance(train_data, torch.Tensor):
                train_data = torch.from_numpy(train_data)
            if not isinstance(train_label, torch.Tensor):
                train_label = torch.from_numpy(train_label)

            train_loader = torch.utils.data.DataLoader(DataSet(train_data, train_label, self.dtype, self.model_type),
                                                       batch_size=self.batch_size, shuffle=self.shuffle)

        if val_loader is None:
            if not isinstance(val_data, torch.Tensor):
                val_data = torch.from_numpy(val_data)
            if not isinstance(val_label, torch.Tensor):
                val_label = torch.from_numpy(val_label)

            val_loader = torch.utils.data.DataLoader(DataSet(val_data, val_label, self.dtype, self.model_type),
                                                     batch_size=self.batch_size, shuffle=self.shuffle)

        return train_loader, val_loader
    
    def add_tb_params(self, ep, tr_mean_loss, val_loss, true, pred, batch_data):
        """
        This function adds tensorboard parameters for training session records.

        Inputs:
            :ep: int, current epoch number.
            :tr_mean_loss: float, average training loss
            :val_loss: float, average validation loss
            :true: torch.Tensor, true labels of validation data
            :pred: torch.Tensor, predicted labels from the model for validation data
            :batch_data: torch.Tensor, last minibatch data.
        """
    	
        if self.tb is not None:
            self.tb.add_scalars('Loss', {'train-loss': tr_mean_loss, 'validation-loss': val_loss}, ep)
            if self.model_type.lower() == 'regressor':
                if self.tb_metrics is None:
                    rsquare, rmse = rsquare_rmse(true.detach().cpu().numpy(), pred.squeeze().detach().cpu().numpy())
                    self.tb.add_scalars('Validation set metrics', {'R-square': rsquare, 'RMSE': rmse}, ep)
                else:
                    for name in self.tb_metrics:
                        score = self.tb_metrics[name](true.detach().cpu().numpy(), pred.squeeze().detach().cpu().numpy())
                        self.tb.add_scalar(name, score, ep)
            elif self.model_type.lower() == 'binary-classifier':
                if self.tb_metrics is None:
                    accuracy = (pred.squeeze().round() == true).sum() / true.shape[0]
                    self.tb.add_scalars('Validation set metrics', {'Accuracy': accuracy}, ep)
                else:
                    for name in self.tb_metrics:
                        score = self.tb_metrics[name](true.detach().cpu().numpy(), pred.squeeze().round().detach().cpu().numpy())
                        self.tb.add_scalar(name, score, ep)
            elif self.model_type.lower() == 'multi-classifier':
                if self.tb_metrics is None:
                    accuracy = (pred.argmax(dim=1).squeeze() == true).sum() / true.shape[0]
                    self.tb.add_scalars('Validation set metrics', {'Accuracy': accuracy}, ep)
                else:
                    for name in self.tb_metrics:
                        score = self.tb_metrics[name](true.detach().cpu().numpy(), pred.argmax(dim=1).squeeze().detach().cpu().numpy())
                        self.tb.add_scalar(name, score, ep)
            if ep == 0:
                self.tb.add_graph(self.model, batch_data.to(device=self.device, dtype=self.dtype))


def instantiate_models():
    """
    This is not a real function, this is a dummy function which shows how a "models" dict looks like.
    This models dict can be used for input arguments for the function "train_with_data"
    """
    models = {
        'Lightgbm': LGBC(),
        'pytorch': torchModel()
    }

    featimp_models = ['RandomForest', 'Xgboost', 'Lightgbm']
    
    return models, featimp_models
