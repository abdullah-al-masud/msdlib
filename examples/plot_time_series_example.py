"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""

import pandas as pd
import numpy as np

import os
import sys
project_dir = os.getcwd()
sys.path.append(project_dir)
from msdlib import msd


"""
We want to see 3 things.
1. Time series data for open, close, volume and percentage_change.
    Here opena and close can be plotted in same graph as their values fluctuate in similar range.
    But volume and percentage change have completely different range of values.
    So, we want to plot them all together to understand their nature using line plots
2. See where the close price goes beyond 95th percentile and keeps at least for 10 days using span plot
3. when the close price and volume was highest and lowest using dotted lines
"""

# loading time series data
# Data is Corn futures data downloaded from here- https://www.investing.com/commodities/us-corn-historical-data

columns = ['date', 'open', 'high', 'low',
           'close', 'volume', 'percentage_change']
ts_data = pd.read_csv(
    'examples/data/US_corn_futures_historical_data.csv').reset_index(drop=True)
ts_data.columns = [c.lower().strip() for c in ts_data.columns]
ts_data = ts_data[columns]
tmp = []
for c in ts_data['volume'].values:
    try:
        tmp.append(float(c))
    except Exception as e:
        try:
            tmp.append(float(c.lower().replace('k', '')) * 1000)
        except Exception as e:
            tmp.append(np.nan)
ts_data['volume'] = tmp
ts_data['percentage_change'] = [
    float(c.replace('%', '')) for c in ts_data['percentage_change'].values]
ts_data['date'] = pd.to_datetime(ts_data['date'])
ts_data = ts_data.set_index('date').sort_index()
print(ts_data.head())
print(ts_data.info())

# calculating the time spans above 95th percentile for close price
close_95th = ts_data['close'].quantile(.95)
print(close_95th)
close_above_95th = msd.get_edges_from_ts(
    ts_data['close'], th=close_95th, del_side='down', name='close_above_95th')
print(close_above_95th)
close_above_95th = close_above_95th[close_above_95th['duration'] >= pd.Timedelta(
    days=10)]

# calculating min and max timestamps for close price and volume
close_mindate = pd.Series(ts_data['close'].idxmin(), name='min_close')
close_maxdate = pd.Series(ts_data['close'].idxmax(), name='max_close')
volume_mindate = pd.Series(ts_data['volume'].idxmin(), name='min_volume')
volume_maxdate = pd.Series(ts_data['volume'].idxmax(), name='max_volume')

print(close_above_95th)
print(close_maxdate)

# variables that will share same y axis
same_srs = [ts_data['open'], ts_data['close']]
# list of pandas series which will share different y axes
srs = [ts_data['volume'], ts_data['percentage_change']]
# line plots for min and max values
lines = [close_mindate, close_maxdate, volume_mindate, volume_maxdate]
# span plots for showing the ranges where the price was above 95th percentile
spans = [close_above_95th]
# number of rows in the plot to understand the curve better
segs = 3
fig_title = 'Combined Time Series Plot'
# plotting simple time series plot for multiple time series variables
msd.plot_time_series(same_srs=same_srs, srs=srs, spans=spans, spine_dist=.05, ylabel='Price',
                     segs=segs, lines=lines, linestyle='--', fig_title=fig_title, save=True,
                     savepath='examples/plot_time_series_example', fname=fig_title.replace(' ', '_'))
