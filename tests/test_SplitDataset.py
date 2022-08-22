"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""

from msdlib.msd.processing import SplitDataset
from sklearn.datasets import load_iris
import pandas as pd


# Loading the data and separating data and label
source_data = load_iris()
feature_names = source_data['feature_names'].copy()
data = pd.DataFrame(source_data['data'], columns=feature_names)
label = pd.Series(source_data['target'], name=source_data['target_names'][1])

splitter = SplitDataset(data, label, test_ratio=.1)


def test_SplitDataset_random_split():

    outdata = splitter.random_split(val_ratio=.15)

    assert outdata['train']['data'].shape[0] == 113
    assert outdata['train']['data'].shape[0] == outdata['train']['label'].shape[0]
    assert outdata['train']['data'].shape[0] == outdata['train']['index'].shape[0]

    assert outdata['validation']['data'].shape[0] == 22
    assert outdata['validation']['data'].shape[0] == outdata['validation']['label'].shape[0]
    assert outdata['validation']['data'].shape[0] == outdata['validation']['index'].shape[0]

    assert outdata['test']['data'].shape[0] == 15
    assert outdata['test']['data'].shape[0] == outdata['test']['label'].shape[0]
    assert outdata['test']['data'].shape[0] == outdata['test']['index'].shape[0]


def test_SplitDataset_sequence_split():

    outdata = splitter.sequence_split(seq_len=5, val_ratio=.15)

    assert outdata['train']['data'].shape[0] == 104
    assert outdata['train']['data'].shape[0] == outdata['train']['label'].shape[0]
    assert outdata['train']['data'].shape[0] == outdata['train']['index'].shape[0]

    assert outdata['validation']['data'].shape[0] == 20
    assert outdata['validation']['data'].shape[0] == outdata['validation']['label'].shape[0]
    assert outdata['validation']['data'].shape[0] == outdata['validation']['index'].shape[0]

    assert outdata['test']['data'].shape[0] == 13
    assert outdata['test']['data'].shape[0] == outdata['test']['label'].shape[0]
    assert outdata['test']['data'].shape[0] == outdata['test']['index'].shape[0]


def test_SplitDataset_CV_split_only_index_True():

    outdata = splitter.cross_validation_split(fold=5, only_index=True)
    
    assert 'data' not in outdata['train']
    assert 'label' not in outdata['train']
    for i in range(1, 6):
        assert outdata['train']['index']['fold_%d'%i].shape[0] == 108
    
    assert 'data' not in outdata['validation']
    assert 'label' not in outdata['validation']
    for i in range(1, 6):
        assert outdata['validation']['index']['fold_%d'%i].shape[0] == 27
    
    assert 'data' not in outdata['test']
    assert 'label' not in outdata['test']

    assert outdata['test']['index'].shape[0] == 15


def test_SplitDataset_CV_split_only_index_False():

    outdata = splitter.cross_validation_split(fold=5, only_index=False)
    
    for i in range(1, 6):
        assert outdata['train']['data']['fold_%d'%i].shape[0] == 108
    for i in range(1, 6):
        assert outdata['validation']['data']['fold_%d'%i].shape[0] == 27

    assert outdata['test']['data'].shape[0] == 15
