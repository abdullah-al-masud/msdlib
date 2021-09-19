"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""

from msdlib import msd
import numpy as np
import pandas as pd


np.random.seed(1216)
# Creating data set for splitting that into Train, Validation and Test
X = pd.DataFrame(np.random.randint(100, size=(1000, 2)), columns=['x', 'y'])
X.index = [pd.Timestamp('20200312010000') + pd.Timedelta(minutes=i * 2)
           for i in range(X.shape[0])]
print('data shape:', X.shape)
print(X)

# Creating labels for the data set
y = pd.Series(np.random.randint(2, size=X.shape[0]), index=X.index)
print('label data\n', y)


# defining object as splitter
splitter = msd.SplitDataset(X, y, test_ratio=.1)

# applying sequence split
outdata = splitter.sequence_split(seq_len=10, val_ratio=.2, data_stride=2, label_shift=3,
                                  split_method='multiple_train_val', sec=2, dist=1)

# Checking output of split function
for _set in outdata.keys():
    if _set == 'test':
        print('%s; data, label and index shapes:' % (_set), outdata[_set]['data'].shape,
              outdata[_set]['index'].shape, outdata[_set]['index'].shape)
    else:
        for _split in outdata['train']['data'].keys():
            print('%s %s ; data, label and index shapes:' % (_set, _split), outdata[_set]['data'][_split].shape,
                  outdata[_set]['index'][_split].shape, outdata[_set]['index'][_split].shape)
