"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""

from msdlib import mlutils
import torch
from msdlib import msd


def test_define_layers():
    """
    To check the define_layers function's activity
    """
    units = [5, 5, 5, 5, 5, 5]
    input_units = 10
    output_units = 2
    activation = torch.nn.ReLU()
    model_type = 'multi-classifier'

    total_layers = 14

    layers = mlutils.define_layers(
        input_units, output_units, units, actual_units=True, model_type=model_type, apply_bn=True)
    assert len(layers) == total_layers


def test_torchModel():

    least_f1_score = .95

    from sklearn.datasets import load_iris

    dataloader = load_iris()
    data = dataloader['data']
    label = dataloader['target']

    splitter = msd.SplitDataset(data, label)
    outdata = splitter.random_split(val_ratio=.2)

    layers = mlutils.define_layers(data.shape[1], len(dataloader['target_names']), [100, 100],
                                   model_type='multi-classification', actual_units=True, activation=torch.nn.ReLU())
    tmodel = mlutils.torchModel(
        layers, epoch=80, model_type='multi-classifier', plot_loss=False, plot_evaluation=False)
    tmodel.fit(outdata['train']['data'], outdata['train']['label'], val_data=outdata['validation']['data'],
               val_label=outdata['validation']['label'], evaluate=False)
    pred = tmodel.predict(outdata['validation']
                          ['data']).argmax(1).cpu().numpy()
    result, distmat = msd.class_result(
        outdata['validation']['label'], pred, True)
    assert result['average'].loc['f1_score'] >= least_f1_score
