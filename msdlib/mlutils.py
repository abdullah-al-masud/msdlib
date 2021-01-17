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
    def __init__(self, layer_funcs, units, activations):
        
        super(NNmodel, self).__init__()
        
        seed_value = 1216
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.total_layers = len(layer_funcs)
        self.activations = activations
        
        self.layers = []
        last_valid = units[0]
        for i in range(self.total_layers):
            if units[i + 1] is not None: 
                self.layers.append(layer_funcs[i](last_valid, units[i + 1]))
                last_valid = units[i + 1]
            else:
                self.layers.append(layer_funcs[i])
        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, x):
        for i in range(self.total_layers):
            x = self.layers[i](x)
            if self.activations[i] is not None: x = self.activations[i](x)
        return x


def train_model(train_data, train_label, test_data, test_label, params):
    # unwrapping parameters
    model = params['model']
    loss_func = params.get('loss_func', nn.MSELoss)
    lr = params.get('learning_rate', .001)
    lr_reduce = params.get('lr_reduce', 1)
    epoch = params.get('epoch', 2)
    reduction = params.get('reduction', 'mean')
    batch_size = params.get('batch_size', 32)
    model_type = params.get('model_type', None)
    use_gpu = params.get('use_gpu', False)
    modelname = params.get('model_name', 'pytorch')

    # getting the model and data ready
    total_batch = train_data.shape[0] // batch_size + int(bool(train_data.shape[0] % batch_size))
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_data = torch.from_numpy(train_data).float().to(device)
    test_data = torch.from_numpy(test_data).float().to(device)
    train_label = torch.from_numpy(train_label).long().to(device)
    test_label = torch.from_numpy(test_label).long().to(device)

    # internal parameter set up
    optimizer = params['optimizer'](model.parameters(), lr = lr)
    loss_func = loss_func(reduction=reduction).to()
    lr_lambda = lambda ep : lr_reduce ** ep
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # running through epoch
    loss_curves = [[], []]
    for ep in range(epoch):
        tr_mean_loss = []
        test_mean_loss = []
        for i in range(total_batch):
            # preparing data set
            if i != total_batch - 1:
                batch_data = train_data[i * batch_size : (i + 1) * batch_size]
                batch_label = train_label[i * batch_size : (i + 1) * batch_size]
            else:
                batch_data = train_data[-batch_size:]
                batch_label = train_label[-batch_size:]
            # run training
            model.train()
            model.zero_grad()
            label_hat = model(batch_data)
            tr_loss = loss_func(label_hat, batch_label)
            # back-propagation
            tr_loss.backward()
            optimizer.step()
            # run evaluation to get validation score
            model.eval()
            out = model(test_data)
            val_loss = loss_func(out, test_label)
            # stacking losses
            tr_mean_loss.append(tr_loss.item())
            test_mean_loss.append(val_loss.item())
            print('\repoch : %04d/%04d, batch : %03d, train_loss : %.4f, validation_loss : %.4f,            '
                  %(ep + 1, epoch, i + 1, tr_loss.item(), val_loss.item()), end = '')
        scheduler.step()
        loss_curves[0].append(np.mean(tr_mean_loss))
        loss_curves[1].append(np.mean(test_mean_loss))
    losses = pd.DataFrame(loss_curves, index = ['train_loss', 'validation_loss'], columns = np.arange(1, epoch + 1)).T.rolling(2).mean()

    # plotting loss curve
    if params['plot_loss'] and epoch > 1:
        ylim_upper = losses.quantile(params.get('quant_perc', .95)).max()
        ylim_lower = losses.min().min()
        fig, ax = plt.subplots(figsize = (25, 4))
        losses.plot(ax = ax, color = ['darkcyan', 'crimson'])
        ax.set_ylim(ylim_lower, ylim_upper)
        fig.suptitle('Learning curves for lr=%.4f, lr_reduce=%.2f'%(lr, lr_reduce), 
                     y = 1, fontsize = 15, fontweight = 'bold')
        fig.tight_layout()
        plt.show()
    # plotting true vs prediction curve (regression) or confusion matrix (classification)
    if params['plot_true_pred'] and model_type.lower() in ['regressor', 'classifier']:
        model.eval()
        setnames = ['Train_set', 'Test_set']
        all_results = {}
        results = []
        for i, (preddata, predlabel) in enumerate(zip([train_data, test_data], [train_label, test_label])):
            test_pred = model(preddata).detach().cpu()
            if model_type == 'regressor':
                true_pred = pd.DataFrame([predlabel, test_pred.numpy()], index = ['true_label', 'prediction']).T
                corr_val = true_pred.corr().iloc[0, 1]
                rsquare, rmse = msd.rsquare_rmse(true_pred['true_label'].values, true_pred['prediction'].values)
                fig, ax = plt.subplots(figsize = (15, 6))
                ax.scatter(true_pred['true_label'], true_pred['prediction'], color = 'darkcyan', s = 8)
                ax.set_xlabel('true-label')
                ax.set_ylabel('prediction')
                ax.set_title('True-Label VS Prediction Scatter plot\nRSquare : %.3f,  RMSE : %.3f,  Correlation : %.3f'
                             %(rsquare, rmse, corr_val))
                all_results[setnames[i]] = [rsquare, rmse]
                results.append(pd.Series([rsquare, rmse], index = ['r_square', 'rmse'], name = '%s_%s'%(modelname, setnames[i])))
            elif model_type == 'classifier':
                test_pred = test_pred.argmax(axis = 1).numpy()
                label = predlabel.detach().cpu().numpy()
                result, confus = msd.class_result(label, test_pred, out_confus = True)
                fig, ax = plt.subplots(figsize = (18, 4), ncols = 2)
                ax[0] = msd.plot_heatmap(result, annotate = True, fmt = '.3f', xrot = 0, vmax = 1, axobj = ax[0], cmap = 'summer', fig_title = 'Score Matrix')
                ax[1] = msd.plot_heatmap(confus, annotate = True, fmt = 'd', xrot = 0, axobj = ax[1], cmap = 'Blues', fig_title = 'Confusion Matrix')
                fig.suptitle('Classification result for %s from %s'%(setnames[i], modelname), fontsize = 15, fontweight = 'bold')
                all_results[setnames[i]] = [result, confus]
                results.append(pd.Series(result.mean().drop('average').to_list() + [result['average'].loc['accuracy']], index = result.drop('average', axis = 1).columns.to_list() + ['average'], name = '%s_%s'%(modelname, setnames[i])))
            fig.tight_layout()
            plt.show()
        results = pd.concat(results, axis = 1, sort = False).T
        
    return model, losses, results, all_results


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


