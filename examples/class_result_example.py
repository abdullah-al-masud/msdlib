"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""

from msdlib import msd
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

# actual labels
true_label = load_iris()['target']
print('true labels:\n', true_label)

# random prediction labels
predicted_label = np.random.randint(3, size=true_label.shape[0])
print('predicted labels:\n', predicted_label)

results, confmat = msd.class_result(
    true_label, predicted_label, out_confus=True)

print('class weights:', pd.Series(
    true_label).value_counts().sort_index().values/true_label.shape[0])
print('classification score:\n', results)
print('confusion matrix:\n', confmat)
