import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from msdlib import msd
import numpy as np
plt.rcParams['figure.facecolor'] = 'white'


# number of hidden layers = n
# number of layer functions inside layer_funcs = n
# number of unit values inside units = n + 1
# number of activation functions inside activations = n
# in case of DROPOUT, BATCHNORM layers, corresponding unit will be None
# Layer function needs to take in two parameters ; in and out units (example- Dense)
# ----------example---------
# layer_funcs = [nn.Linear, nn.Dropout(.5), nn.Linear]
# units = [train_data.shape[1], 200, None, train_label.shape[1]]
# activations = [F.relu, None, F.softmax]
class NNmodel(nn.Module):
    def __init__(self, layer_funcs):
        
        super(NNmodel, self).__init__()
        # reproducibility parameters
        seed_value = 1216
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.total_layers = len(layer_funcs)
        self.layers = nn.ModuleList(layer_funcs)
        
    def forward(self, x):
        for i in range(self.total_layers):
            x = self.layers[i](x)
        return x
    

# auto-encoder model
# number of unit values inside units = [input_data_shape] + [encoder units] + [decoder units]
# layer_func by default is Dense
class AutoEncoderModel(nn.Module):
    def __init__(self, units, encode_acts, decode_acts, layer_func = nn.Linear):
        super(NNmodel, self).__init__()
        self.layers = len(units) // 2 - 1
        self.encode_layers = []
        self.decode_layers = []
        for i in range(self.layers):
            self.encode_layers.append(layer_func(units[i], units[i + 1]))
            self.decode_layers.append(layer_func(units[self.layers + i], units[self.layers + i + 1]))
        self.encode_layers = nn.ModuleList(self.encode_layers)
        self.decode_layers = nn.ModuleList(self.decode_layers)
        self.encode_acts = encode_acts
        self.decode_acts = decode_acts
    
    def encode(self, x):
        for i in range(self.layers):
            x = self.encode_layers[i](x)
            if self.encode_acts[i] is not None : x = self.encode_acts[i](x)
        return x
    
    def decode(self, x):
        for i in range(self.layers):
            x = self.decode_layers[i](x)
            if self.decode_acts[i] is not None : x = self.decode_acts[i](x)
        return x
        
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


# torchModel is a scikit like wrapper for pytorch which enables us to use the model for 
# training, predicting and evaluating performance using simple fit, predict and evaluate methods
class torchModel():
    
    def __init__(self, layers, loss_func=None, optimizer=None, learning_rate=.0001, epoch=2, batch_size=32, lr_reduce=1, 
                 loss_reduction='mean', model_type='regressor', use_gpu=True, model_name='pytorch', dtype=torch.float32,
                 plot_loss=True, quant_perc=.98, plot_true_pred=True, loss_roll_period=1):
        
        # defining model architecture
        self.model = NNmodel(layers)
        self.loss_func = loss_func if loss_func is not None else nn.MSELoss
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam
        
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
        
        # evaluation parameters
        self.plot_loss = plot_loss
        self.quant_perc = quant_perc
        self.plot_true_pred = plot_true_pred
        self.loss_roll_period = loss_roll_period
        
        # setting up
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.loss_func = loss_func(reduction=self.loss_reduction).to(device=self.device, dtype=self.dtype)
        self.optimizer = self.optimizer(self.model.parameters(), lr = self.learning_rate)
        
        # learning rate scheduler
        lr_lambda = lambda ep : self.lr_reduce ** ep
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
    
    def fit(self, data, label, validation_ratio=.15, evaluate=True, figsize=(15, 6)):
        """
        scikit like wrapper for training DNN model
        data: input train data, must be torch tensor or numpy ndarray
        label: supervised labels for data, must be torch tensor or numpy ndarray
        validation_ratio: ratio of 'data' that will be used for validation during training
        """
        
        # allowing pandas dataframe or series input
        if isinstance(data, pd.DataFrame): data = data.values.copy()
        elif isinstance(data, pd.Series): data = data.to_frame().values.copy()
        if isinstance(label, pd.DataFrame): label = label.values.copy()
        elif isinstance(label, pd.Series): label = label.values.copy()
        
        # splitting data set
        train_ratio = 1 - validation_ratio
        idx = np.arange(label.shape[0])
        np.random.shuffle(idx)
        train_idx = idx[:int(train_ratio * label.shape[0])]
        val_idx = idx[int(train_ratio * label.shape[0]):]
        
        # getting the model and data ready
        total_batch = train_idx.shape[0] // self.batch_size + int(bool(train_idx.shape[0] % self.batch_size))
        
        train_data = data[train_idx]
        train_label = label[train_idx]
        val_data = data[val_idx]
        val_label = label[val_idx]
        
        # handling data sets and labels
        if not isinstance(train_data, torch.Tensor): train_data = torch.from_numpy(train_data)
        if not isinstance(val_data, torch.Tensor): val_data = torch.from_numpy(val_data)
        if not isinstance(train_label, torch.Tensor): train_label = torch.from_numpy(train_label)
        if not isinstance(val_label, torch.Tensor): val_label = torch.from_numpy(val_label)
        
        # data type conversion
        train_data = train_data.to(device=self.device, dtype=self.dtype)
        train_label = train_label.to(device=self.device, dtype=self.dtype) if self.model_type == 'regressor' else train_label.to(device=self.device, dtype=torch.long)
        val_data = val_data.to(device=self.device, dtype=self.dtype)
        val_label = val_label.to(device=self.device, dtype=self.dtype) if self.model_type == 'regressor' else val_label.to(device=self.device, dtype=torch.long)
        
        # running through epoch
        loss_curves = [[], []]
        val_loss = torch.tensor(np.nan)
        for ep in range(self.epoch):
            tr_mean_loss = []
            self.model.train()
            for i in range(total_batch):
                # preparing data set
                if i != total_batch - 1:
                    batch_data = train_data[i * self.batch_size : (i + 1) * self.batch_size]
                    batch_label = train_label[i * self.batch_size : (i + 1) * self.batch_size]
                else:
                    batch_data = train_data[-self.batch_size:]
                    batch_label = train_label[-self.batch_size:]
                
                # loss calculation
                self.model.zero_grad()
                label_hat = self.model(batch_data).squeeze()
                tr_loss = self.loss_func(label_hat, batch_label)
                
                # back-propagation
                tr_loss.backward()
                # model parameter update
                self.optimizer.step()
                
                # stacking and printing losses
                tr_mean_loss.append(tr_loss.item())
                print('\repoch : %04d/%04d, batch : %03d, train_loss : %.4f, validation_loss : %.4f,            '
                      % (ep + 1, self.epoch, i + 1, tr_loss.item(), val_loss.item()), end = '')
            
            # loss scheduler step
            self.scheduler.step()
            
            # run evaluation to get validation score
            self.model.eval()
            out = self.predict(val_data).squeeze()
            val_loss = self.loss_func(out, val_label)
            
            # storing losses
            loss_curves[0].append(np.mean(tr_mean_loss))
            loss_curves[1].append(val_loss.item())
        
        print('...training complete !!')
        losses = pd.DataFrame(loss_curves, index = ['train_loss', 'validation_loss'], columns = np.arange(1, self.epoch + 1)).T.rolling(self.loss_roll_period).mean()
        
        # model training evaluation
        if evaluate: self.evaluate([train_data, val_data], [train_label, val_label], set_names=['Train_set', 'Validation_set'], losses=losses, figsize=figsize)
        
        return losses
    
    
    def predict(self, data):
        """
        a wrapper function that generates prediction from pytorch model
        data: input data to predict on, must be a torch tensor or numpy ndarray
        """
        
        # checking data type
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series): data = data.values
        if isinstance(data, np.ndarray): data = torch.from_numpy(data).to(device=self.device, dtype=self.dtype)
        
        # estimating number of mini-batch
        n_batch = data.shape[0] // self.batch_size + int(bool(data.shape[0] % self.batch_size))
        # generates prediction
        preds = []
        for i in range(n_batch):
            if i != n_batch - 1:
                pred = self.model(data[i * self.batch_size: (i + 1) * self.batch_size])
            else:
                pred = self.model(data[i * self.batch_size:])
            preds.append(pred.detach())
        preds = torch.cat(preds)
        return preds
    
    
    def evaluate(self, data_sets, label_sets, set_names=[], losses=None, figsize=(18, 4)):
        """
        a customized function to evaluate model performance in regression and classification type tasks
        data_sets: list of data, data must be nunmpy ndarray or torch tensor
        label_sets: list of labels corresponding to each data, label must be nunmpy ndarray or torch tensor
        set_names: names of the data sets
        losses: pandas dataFrame containing training and validation loss
        figsize: figure size for the evaluation plots
        """
        # plotting loss curve
        if self.plot_loss and self.epoch > 1 and losses is not None:
            ylim_upper = losses.quantile(params.get('quant_perc', .98)).max()
            ylim_lower = losses.min().min()
            fig, ax = plt.subplots(figsize = (25, 4))
            losses.plot(ax = ax, color = ['darkcyan', 'crimson'])
            ax.set_ylim(ylim_lower, ylim_upper)
            fig.suptitle('Learning curves', y = 1, fontsize = 15, fontweight = 'bold')
            fig.tight_layout()
            plt.show()

        # plotting true vs prediction curve (regression) or confusion matrix (classification)
        if self.plot_true_pred and self.model_type.lower() in ['regressor', 'classifier'] and len(data_sets) > 0:
            set_names = set_names if len(set_names) > 0 else ['data-%d'%(i+1) for i in range(len(data_sets))]
            all_results = {}
            results = []
            self.model.eval()
            for i, (preddata, predlabel) in enumerate(zip(data_sets, label_sets)):
                test_pred = self.predict(preddata).detach().cpu().squeeze().numpy()
                label = predlabel.detach().cpu().squeeze().numpy()
                if self.model_type == 'regressor':
                    true_pred = pd.DataFrame([label, test_pred], index = ['true_label', 'prediction']).T
                    corr_val = true_pred.corr().iloc[0, 1]
                    rsquare, rmse = msd.rsquare_rmse(true_pred['true_label'].values, true_pred['prediction'].values)
                    fig, ax = plt.subplots(figsize=figsize)
                    ax.scatter(true_pred['true_label'], true_pred['prediction'], color = 'darkcyan', s = 8)
                    _min = np.min([true_pred['true_label'].min(), true_pred['prediction'].min()])
                    _max = np.max([true_pred['true_label'].max(), true_pred['prediction'].max()])
                    ax.plot([_min, _max], [_min, _max], color='k', lw=2)
                    print(_min, _max)
                    ax.set_xlabel('true-label')
                    ax.set_ylabel('prediction')
                    ax.set_title('True-Label VS Prediction Scatter plot for %s from %s\nRSquare : %.3f,  RMSE : %.3f,  Correlation : %.3f'
                                 %(set_names[i], self.model_name, rsquare, rmse, corr_val))
                    all_results[set_names[i]] = [rsquare, rmse]
                    results.append(pd.Series([rsquare, rmse], index = ['r_square', 'rmse'], name = '%s_%s'%(self.model_name, set_names[i])))
                elif self.model_type == 'classifier':
                    test_pred = np.argmax(test_pred, axis=1)
                    result, confus = msd.class_result(label, test_pred, out_confus = True)
                    fig, ax = plt.subplots(figsize=figsize, ncols = 2)
                    ax[0] = msd.plot_heatmap(result, annotate = True, fmt = '.3f', xrot = 0, vmax = 1, axobj = ax[0], cmap = 'summer', fig_title = 'Score Matrix')
                    ax[1] = msd.plot_heatmap(confus, annotate = True, fmt = 'd', xrot = 0, axobj = ax[1], cmap = 'Blues', fig_title = 'Confusion Matrix')
                    fig.suptitle('Classification result for %s from %s'%(set_names[i], self.model_name), fontsize = 15, fontweight = 'bold')
                    all_results[set_names[i]] = [result, confus]
                    results.append(pd.Series(result.mean().drop('average').to_list() + [result['average'].loc['accuracy']], index = result.drop('average', axis = 1).columns.to_list() + ['average'], 
                                             name = '%s_%s'%(self.model_name, set_names[i])))
                fig.tight_layout()
                plt.show()
            results = pd.concat(results, axis = 1, sort = False).T

        return results, all_results