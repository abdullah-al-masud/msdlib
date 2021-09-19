"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""

from msdlib import msd
import pandas as pd
from sklearn.datasets import load_iris


# Creating data set for splitting that into Train, Validation and Test
loader = load_iris()
data = pd.DataFrame(loader['data'], columns=loader['feature_names'])
label = pd.Series(loader['target'])
print(data)

# Splitting object
# defining object as splitter
splitter = msd.SplitDataset(data, label, same_ratio=True, test_ratio=.1)


print('applying random split...')
outdata = splitter.random_split(val_ratio=.2)
print(outdata.keys())
print(outdata['train'].keys())
for _set in outdata:
    print('%s data, label and index shapes:' % _set,
          outdata[_set]['data'].shape, outdata[_set]['label'].shape, outdata[_set]['index'].shape)


print('\n\n applying cross validation split...')
outdata = splitter.cross_validation_split(fold=5)
print(outdata.keys())
print(outdata['train'].keys())
for _set in outdata:
    if _set == 'test':
        print('%s data, label and index shapes:' % _set,
              outdata[_set]['data'].shape, outdata[_set]['label'].shape, outdata[_set]['index'].shape)
    else:
        for fold in outdata[_set]['label'].keys():
            print('fold=%s; %s data, label and index shapes:' % (fold, _set),
                  outdata[_set]['data'][fold].shape, outdata[_set]['label'][fold].shape, outdata[_set]['index'][fold].shape)
