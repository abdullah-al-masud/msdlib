"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""


import math
import pandas as pd


# loading data
def load_data(root_path, file, start=None, stop=None, usables=['open', 'high', 'low', 'close', 'volume']):
    """
    This function loads csv or txt formatted data. 

    Inputs:
        :root_path: path to the directory where the csv or txt file is kept
        :file: file name of the csv or txt file with extension
        :start: starting time for the data, default is None
        :stop: ending time for the data, default is None
        :usables: column names in lower case letter which will be used from csv or txt data. Default columns are [open, high, low, close, volume]

    Outputs:
        :data: loaded data
    """
    usables = [c.lower() for c in usables]
    data = pd.read_csv('%s/%s' % (root_path, file)).set_index('Date')
    if start is None:
        start = data.index[0]
    if stop is None:
        stop = data.index[-1]
    data.columns = [c.lower().replace(' ', '') for c in data.columns]
    data.index = [pd.Timestamp(i) for i in data.index]
    data = data[usables].loc[start: stop]
    return data

# getting crossover points


def get_crossover(sr_slow, sr_fast):
    """
    This functions calculates crossover points between a faster time series (sr_fast) and a bit slower version of the same time series data.

    Inputs:
        :sr_slow: pandas Series, slow time series data (rolling average with bigger window size of a time series data)
        :sr_fast: pandas Series, faster time series data (rolling average with bigger smaller size of a time series data)

    Outputs:
        :cross_sig: pandas Series, containing crossover points 1 (fast crosses slow signal from down to up) or -1 (slow crosses fast signal from up to down) 
    """
    srfast_prev = sr_fast.shift(1)
    srslow_prev = sr_slow.shift(1)
    high_cross = sr_fast[(srfast_prev <= srslow_prev)
                         & (sr_fast > sr_slow)].index
    low_cross = sr_fast[(srfast_prev >= srslow_prev)
                        & (sr_fast < sr_slow)].index
    cross_sig = pd.Series(0, index=sr_fast.index, name='crossover')
    cross_sig.loc[high_cross] = 1
    cross_sig.loc[low_cross] = -1
    return cross_sig

# getting next available dates


def next_dates(datetimes, dates):
    """
    gets next date of a particular series data from pandas series index.

    Inputs:
        :datetimes: list of datetimes, to find the next dates of
        :dates: list of indices of true time series data

    Outputs:
        :nextdates: list of next dates corresponding to datetimes 
    """
    _len = len(datetimes)
    lendate = len(dates)
    cnt = 0
    if len(dates) > 0:
        date = dates[0]
        nextdates = []
        for i in range(_len):
            if date == datetimes[i] and i != _len - 1:
                nextdates.append(datetimes[i + 1])
                cnt += 1
                if cnt >= lendate:
                    break
                date = dates[cnt]
    else:
        nextdates = []
    return nextdates

# getting previous available dates


def prev_dates(datetimes, dates):
    """
    gets predious date of a particular series data from pandas series index.

    Inputs:
        :datetimes: list of datetimes, to find the previous dates of
        :dates: list of indices of true time series data

    Outputs:
        :prevdates: list of next dates corresponding to datetimes 
    """
    datetimes = list(datetimes)
    dates = list(dates) if not (isinstance(dates, str)
                                or isinstance(dates, pd.Timestamp)) else [dates]
    _len = len(datetimes)
    lendate = len(dates)
    cnt = 0
    if len(dates) > 0:
        date = dates[0]
        prevdates = []
        for i in range(_len):
            if date == datetimes[i] and i != 0:
                prevdates.append(datetimes[i - 1])
                cnt += 1
                if cnt >= lendate:
                    break
                date = dates[cnt]
    else:
        prevdates = []
    return prevdates

# getting balance


def get_balance(ts_index, closings, starting_cap, sample_per='D'):
    """
    This function calcualtes total available balance for a portfolio.

    Inputs:
        :ts_index: index of time series pandas data
        :closings: pandas dataframe, having columns 'trigger_point', 'rated_return'\n
            'trigger_point' column contains time stamp when there is buy or sell signal found\n
            'rated_return' column contains gain(+) or loss(-) values corresponding to each trigger_point
        :starting_cap: float, starting capital value
        :sample_per: Literals like 'D', 'M', 'H' etc., resampling period, default is 'D'

    Outputs:
        balance: pandas Series, containing balance value for a portfolio or one symbol for each time instance (balance curve)
    """
    balance = pd.Series(math.nan, index=ts_index,
                        name='balance').resample(sample_per).asfreq()
    balance.loc[:closings['trigger_point'].iloc[0] -
                pd.Timedelta(days=1)] = starting_cap
    balance.loc[closings['trigger_point']] = (
        closings['rated_return'].cumsum() + starting_cap).values
    balance.fillna(method='ffill', inplace=True)
    return balance

# calculate drawdown dataframe


def get_drawdown(sr):
    """
    This function calculates drawdown value from a balance curve

    Inputs:
        :sr: padnas Series, time series balance at each time instance (balance curve)

    Outputs:
        :dds: pandas DataFrame, contains these columns ['start_point', 'start_value', 'low_point', 'low_value', 'end_point', 'end_value', 'duration']
    """
    cumax = sr.cummax()
    curval = cumax.iloc[0]
    lastidx = 0
    dds = []
    for i in range(1, sr.shape[0]):
        if cumax.iloc[i] > curval:
            if i - lastidx > 1:
                minidx = sr.iloc[lastidx + 1: i].idxmin()
                minval = sr.loc[minidx]
                if minval < curval:
                    duration = sr.index[i] - sr.index[lastidx]
                    dds.append([sr.index[lastidx], sr.iloc[lastidx], minidx,
                               minval, sr.index[i - 1], sr.iloc[i - 1], duration])
            curval = cumax.iloc[i]
            lastidx = i
        elif sr.iloc[i] == curval:
            lastidx = i
    if sr.iloc[-1] < curval:
        minidx = sr.iloc[lastidx + 1:].idxmin()
        minval = sr.loc[minidx]
        duration = sr.index[-1] - sr.index[lastidx]
        dds.append([sr.index[lastidx], sr.iloc[lastidx], minidx,
                   minval, sr.index[-1], sr.iloc[-1], duration])
    if len(dds) > 1:
        dds = pd.DataFrame(dds, columns=[
                           'start_point', 'start_value', 'low_point', 'low_value', 'end_point', 'end_value', 'duration'])
    else:
        dds = [math.nan, sr.iloc[0], math.nan, sr.iloc[0],
               math.nan, sr.iloc[-1], pd.Timedelta(seconds=0)]
        dds = pd.DataFrame([dds], columns=['start_point', 'start_value',
                           'low_point', 'low_value', 'end_point', 'end_value', 'duration'])
    dds['drawdown%'] = (1 - dds['low_value'] / dds['start_value']) * 100
    dds['drawdown%'][dds['drawdown%'] < 0] = 0
    return dds

# calculate calmar ratio


def get_calmar(sr):
    """
    This function calculates calmar ratio value for a balance curve

    Inputs:
        :sr: pandas Series, balance time series data (balance curve)

    Outputs:
        :calmar: float, calmar ratio value
    """
    drawdown = get_drawdown(sr)
    max_drawdown = drawdown['drawdown%'].max()
    annual_return = get_annual_return(sr)
    calmar = annual_return / max_drawdown
    return calmar

# calculate compound annual return


def get_annual_return(sr):
    """
    This function calculates annual rate of return

    Inputs: 
        :sr: pandas Series, balance curve

    Outputs:
        :annual_return: float, rate of return value

    """
    annual_returns = sr.resample('Y').last() / sr.resample('Y').first()
    annual_return = (annual_returns.cumprod(
    ).iloc[-1] ** (1 / annual_returns.shape[0]) - 1) * 100
    return annual_return
