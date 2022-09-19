"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.signal import butter, sosfiltfilt, sosfreqz
from scipy.signal import spectrogram as spect
from ..msdExceptions import (
    CutoffError,
    FilterTypeError,
    InvalidBins,
    InputVariableError,
    DataLabelMismatchError,
    SplitMethodError
)
import os
import warnings

sns.set()
pd.plotting.register_matplotlib_converters()


def get_time_estimation(time_st, count_ratio=None, current_ep=None, current_batch=None, total_ep=None, total_batch=None, string_out=True):
    """
        This function estimates remaining time inside any loop. he function is prepared for estimating 
        remaining time in machine learning training with mini-batch.
        But it can be used for other purposes also by providing count_ratio input value.
        
        Inputs:
            :time_st: time.time() instance indicating the starting time count
            :count_ratio: float, ratio of elapsed time at any moment.\n
                          Must be 0 ~ 1, where 1 will indicate that this is the last iteration of the loop
            :current_ep: current epoch count
            :current_batch: current batch count in mini-batch training
            :total_ep: total epoch in the training model
            :total_batch: total batch in the mini-batch training
            :string_out: bool, whether to output the elapsed and estimated time in string format or not. 
                        True will output time string
        
        Outputs:
            :output time: the ouput can be a single string in format\n
                        'elapsed hour : elapsed minute : elapsed second < remaining hour : remaining minute : remaining second '\n
                        if string_out flag is True\n
                        or it can output 6 integer values in the above order for those 6 elements in the string format\n
    """
    if count_ratio is None:
        # getting count ratio
        total_count = total_ep * total_batch
        current_count = current_ep * total_batch + current_batch + 1
        count_ratio = current_count / total_count

    # calculating time
    t = time.time()
    elapsed_t = t - time_st

    # converting time into H:M:S
    # elapsed time calculation
    el_h = elapsed_t // 3600
    el_m = (elapsed_t - el_h * 3600) // 60
    el_s = int(elapsed_t - el_h * 3600 - el_m * 60)

    if count_ratio == 0:
        if string_out:
            return '%d:%02d:%02d < %d:%02d:%02d' % (el_h, el_m, el_s, 0, 0, 0)
        else:
            return el_h, el_m, el_s, 0, 0, 0

    # remaining time calculation
    total_t = elapsed_t / count_ratio
    rem_t = total_t - elapsed_t
    rem_h = rem_t // 3600
    rem_m = (rem_t - rem_h * 3600) // 60
    rem_s = int(rem_t - rem_h * 3600 - rem_m * 60)

    if string_out:
        out = '%d:%02d:%02d < %d:%02d:%02d' % (el_h, el_m, el_s, rem_h, rem_m, rem_s)
        return out
    else:
        return el_h, el_m, el_s, rem_h, rem_m, rem_s


class Filters():
    """ 
    This class is used to apply FIR filters as high-pass, low-pass, band-pass and band-stop filters.
    It also shows the proper graphical representation of the filter response, signal before filtering and after filtering and filter spectram.
    The core purpose of this class is to make the filtering process super easy and check filter response comfortably.
    
    Inputs:
        :T: float, indicating the sampling period of the signal, must be in seconds (doesnt have any default values)
        :n: int, indicating number of fft frequency bins, default is 1000
        :savepath: str, path to the directory to store the plot, default is None (doesnt save plots)
        :show: bool, whether to show the plot or not, default is True (Shows figures)
        :save: bool, whether to store the plot or not, default is False (doesnt save figures)
    """
    
    def __init__(self, T, N = 1000, savepath=None, save=False, show=True):
        # T must be in seconds
        self.fs = 1 / T
        self.N = N
        self.savepath = savepath
        self.show = show
        self.save = save
   
    def raise_cutoff_error(self, msg):
        raise CutoffError(msg)
   
    def raise_filter_error(self, msg):
        raise FilterTypeError(msg)
   
   
    def vis_spectrum(self, sr, f_lim=[], see_neg=False, show=None, save=None, savepath=None, figsize=(30, 3)):
        """
        The purpose of this function is to produce spectrum of time series signal 'sr'. 
        
        Inputs:
            :sr: numpy ndarray or pandas Series, indicating the time series signal you want to check the spectrum for
            :f_lim: python list of len 2, indicating the limits of visualizing frequency spectrum. Default is []
            :see_neg: bool, flag indicating whether to check negative side of the spectrum or not. Default is False
            :show: bool, whether to show the plot or not, default is None (follows Filters.show attribute)
            :save: bool, whether to store the plot or not, default is None (follows Filters.save attribute)
            :savepath: str, path to the directory to store the plot, default is None (follows Filters.savepath attribute)
            :figsize: tuple, size of the figure plotted to show the fft version of the signal. Default is (30, 3)
        
        Outputs:
            doesnt return anything as the purpose is to generate plots
        """

        if savepath is None: savepath = self.savepath
        if save is None: save = self.save
        if show is None: show = self.show

        if isinstance(sr, np.ndarray) or isinstance(sr, list): sr = pd.Series(sr)
        if sr.name is None: sr.name = 'signal'
        if see_neg:
            y = np.fft.fft(sr.dropna().values, n = self.N).real
            y = np.fft.fftshift(y)
            f = (np.arange(self.N) / self.N - .5) * self.fs
            y = pd.Series(y, index = f)
        else:
            y = np.fft.fft(sr.dropna().values, n = self.N * 2).real
            f = np.arange(2 * self.N) / (2 * self.N) * self.fs
            y = pd.Series(y, index = f)
            y = y.iloc[:self.N]
        fig, ax = plt.subplots(figsize = figsize)
        ax.plot(y.index, y.values, alpha = 1)
        fig_title = 'Frequency Spectrum of %s'%sr.name
        ax.set_title(fig_title)
        ax.set_ylabel('Power')
        ax.set_xlabel('Frequency (Hz)')
        if len(f_lim) == 2: ax.set_xlim(f_lim[0], f_lim[1])
        fig.tight_layout()
        if show:
            plt.show()
        if save and savepath is not None:
            os.makedirs(savepath, exist_ok=True)
            fig.savefig('%s/%s.jpg'%(savepath, fig_title.replace(' ', '_')), bbox_inches='tight')
        plt.close()
   
    def apply(self, sr, filt_type, f_cut, order=10, response=False, plot=False, f_lim=[], savepath=None, show=None, save=None):
        """
        The purpose of this function is to apply an FIR filter on a time series signal 'sr' and get the output filtered signal y

        Inputs:
            :sr: numpy ndarray/pandas Series/list, indicating the time series signal you want to apply the filter on
            :filt_type: str, indicating the type of filter you want to apply on sr.\n
                        {'lp', 'low_pass', 'low pass', 'lowpass'} for applying low pass filter\n
                        and similar for 'high pass', 'band pass' and 'band stop' filters
            :f_cut: float/list/numpy 1d array, n indicating cut off frequencies. Please follow the explanations bellow\n
                    for lowpass/highpass filters, it must be int/float\n
                    for bandpass/bandstop filters, it must be a list or numpy array of length divisible by 2
            :order: int, filter order, the more the values, the sharp edges at the expense of complexer computation. Default is 10.
            :response: bool, whether to check the frequency response of the filter or not, Default is False
            :plot: bool, whether to see the spectrum and time series plot of the filtered signal or not, Default is False
            :f_lim: list of length 2, frequency limit for the plot. Default is []
            :savepath: str, path to the directory to store the plot, default is None (follows Filters.savepath attribute)
            :show: bool, whether to show the plot or not, default is None (follows Filters.show attribute)
            :save: bool, whether to store the plot or not, default is None (follows Filters.save attribute)
        
        Outputs:
            :y: pandas Series, filtered output signal
        """
        
        if savepath is None: savepath = self.savepath
        if save is None: save = self.save
        if show is None: show = self.show
        if isinstance(sr, np.ndarray) or isinstance(sr, list): sr = pd.Series(sr)
        if sr.name is None: sr.name = 'signal'
        msg = 'Cut offs should be paired up (for bp or bs) or in integer format (for hp or lp)'
        if isinstance(f_cut, list) or isinstance(f_cut, np.ndarray):
            if len(f_cut) % 2 != 0:
                self.raise_cutoff_error(msg)
            else:
                band = True
                f_cut = np.array(f_cut).ravel() / (self.fs / 2)
                if np.sum(f_cut <= 0) > 0:
                    msg = 'Invalid cut offs (0 or negative cut offs)'
                    self.raise_cutoff_error(msg)
                elif np.sum(f_cut > 1) > 0:
                    msg = 'Invalid cut offs (frequency is greater than nyquest frequency)'
                    self.raise_cutoff_error(msg)
        elif isinstance(f_cut, int) or isinstance(f_cut, float):
            band = False
            f_cut = f_cut / (self.fs / 2)
            if f_cut <= 0:
                msg = 'Invalid cut offs (0 or negative cut offs)'
                self.raise_cutoff_error(msg)
            elif f_cut > 1:
                msg = 'Invalid cut offs (frequency is greater than nyquest frequency)'
                self.raise_cutoff_error(msg)
        else:
            self.raise_cutoff_error(msg)
       
        msg = 'Filter type %s is not understood or filter_type-cutoff mis-match'%(filt_type)
        filt_type = filt_type.lower()
        if filt_type in ['low_pass', 'lp', 'low pass', 'lowpass'] and not band:
            _filter = butter(N = order, Wn = f_cut, btype = 'lowpass', analog = False, output = 'sos')
        elif filt_type in ['high_pass', 'hp', 'high pass', 'highpass'] and not band:
            _filter = butter(N = order, Wn = f_cut, btype = 'highpass', analog = False, output = 'sos')
        elif filt_type in ['band_pass', 'bp', 'band pass', 'bandpass'] and band:
            _filter = butter(N = order, Wn = f_cut, btype = 'bandpass', analog = False, output = 'sos')
        elif filt_type in ['band_stop', 'bs', 'band_stop', 'bandstop'] and band:
            _filter = butter(N = order, Wn = f_cut, btype = 'bandstop', analog = False, output = 'sos')
        else:
            self.raise_filter_error(msg)
        y = pd.Series(sosfiltfilt(_filter, sr.dropna().values), index = sr.dropna().index, name = 'filtered_%s'%sr.name)
        if response:
            self.filter_response(_filter, f_lim = f_lim, savepath=savepath, save=save, show=show)
        if plot:
            self.plot_filter(y, filt_type, f_cut, f_lim = f_lim, savepath=savepath, save=save, show=show)
        return y
   
   
    def filter_response(self, sos, f_lim=[], savepath=None, show=None, save=None):
        """
        This function plots the filter spectram in frequency domain.

        Inputs:
            :sos: sos object found from butter function
            :flim: list/tuple as [lower_cutoff, higher_cutoff], default is []
            :savepath: str, path to the directory to store the plot, default is None (follows Filters.savepath attribute)
            :show: bool, whether to show the plot or not, default is None (follows Filters.show attribute)
            :save: bool, whether to store the plot or not, default is None (follows Filters.save attribute)
        
        Outputs:
            It doesnt return anything.

        """
        if savepath is None: savepath = self.savepath
        if save is None: save = self.save
        if show is None: show = self.show

        w, h = sosfreqz(sos, worN = 2000)
        w = (self.fs / 2) * (w / np.pi)
        h = np.abs(h)
        
        fig, ax = plt.subplots(figsize = (30, 3))
        ax.plot(w, h)
        ax.set_title('Filter spectrum')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Gain')
        if f_lim != []: ax.set_xlim(f_lim)
        fig.tight_layout()
        if show: plt.show()
        if savepath is not None and save:
            os.makedirs(savepath, exist_ok=True)
            fig.savefig(os.path.join(savepath, 'Filter_spectrum.jpg'), bbox_inches='tight')
        plt.close()
       
    def plot_filter(self, y, filt_type, f_cut, f_lim=[], savepath=None,  show=None, save=None):
        """
        This function plots filter frequency response, time domain signal etc.

        Inputs:
            :y: pandas Series, time series data to filter
            :filt_type: str, indicating the type of filter you want to apply on y.\n
                       {'lp', 'low_pass', 'low pass', 'lowpass'} for applying low pass filter\n 
                       and similar for 'high pass', 'band pass' and 'band stop' filters
            :f_cut: float/list/numpy 1d array, n indicating cut off frequencies. Please follow the explanations bellow\n
                    for lowpass/highpass filters, it must be int/float\n
                    for bandpass/bandstop filters, it must be a list or numpy array of length divisible by 2
            :f_lim: list of length 2, frequency limit for the plot. Default is []
            :savepath: str, path to the directory to store the plot, default is None (follows Filters.savepath attribute)
            :show: bool, whether to show the plot or not, default is None (follows Filters.show attribute)
            :save: bool, whether to store the plot or not, default is None (follows Filters.save attribute)
            
        Outputs:
            No output is returned. The function generates plot.
        """

        if savepath is None: savepath = self.savepath
        if save is None: save = self.save
        if show is None: show = self.show

        title = '%s with %s filter with cut offs %s'%(y.name, filt_type, str(f_cut * (self.fs / 2)))
        self.vis_spectrum(sr=y, f_lim=f_lim, show=show)
        fig, ax = plt.subplots(figsize=(30, 3))
        ax.set_title(title)
        ax.plot(y.index, y.values)
        ax.set_xlabel('Time')
        ax.set_ylabel('%s'%y.name)
        fig.tight_layout()
        if show: plt.show()
        if save and savepath is not None:
            os.makedirs(savepath, exist_ok=True)
            fig.savefig(os.path.join(savepath, title.replace(' ', '_')+'.jpg'), bbox_inches='tight')
        plt.close()


def get_spectrogram(ts_sr, fs=None, win=('tukey', 0.25), nperseg=None, noverlap=None, mode='psd', figsize=None, 
                    vis_frac=1, ret_sxx=False, show=True, save=False, savepath=None, fname=None):
    """
    The purpose of this function is to find the spectrogram of a time series signal. 
    It computes spectrogram, returns the spectrogram output and also i able to show spectrogram plot as heatmap.

    Inputs:
        :ts_sr: pandas.Series object containing the time series data, the series should contain its name as ts_sr.name
        :fs: int/flaot, sampling frequency of the signal, default is 1
        :win: tuple of (str, float), tuple of window parameters, default is (tukey, .25)
        :nperseg: int, number of data in each segment of the chunk taken for STFT, default is 256
        :noverlap: int, number of data in overlapping region, default is (nperseg // 8)
        :mode: str, type of the spectrogram output, default is power spectral density('psd')
        :figsize: tuple, figsize of the plot, default is (30, 6)
        :vis_frac: float or a list of length 2, fraction of the frequency from lower to higher, you want to visualize, default is 1(full range)
        :ret_sxx: bool, whehter to return spectrogram dataframe or not, default is False
        :show: bool, whether to show the plot or not, default is True
        :save: bool, whether to save the figure or not, default is False
        :savepath: str, path to save the figure, default is None
        :fname: str, name of the figure to save in the savepath, default is figure title
    
    Outputs:
        :sxx: returns spectrogram matrix if ret_sxx flag is set to True
    """

    # default values
    if fs is None: fs = 1
    if nperseg is None: nperseg = 256
    if noverlap is None: noverlap = nperseg // 8
    if figsize is None: figsize = (30, 6)
    if isinstance(vis_frac, int) or isinstance(vis_frac, float): vis_frac = [0, vis_frac]
   
    # applying spectrogram from scipy.signal.spectrogram
    f, t, sxx = spect(x = ts_sr.values, fs = fs, window = win, nperseg = nperseg, noverlap = noverlap, mode = mode)
#     f = ['%.6f'%i for i in f]
    # properly spreading the time stamps
    t = [ts_sr.index[i] for i in (ts_sr.shape[0] / t[-1] * t).astype(int) - 1]
    # creating dataframe of spectrogram
    sxx = pd.DataFrame(sxx, columns = t, index = f)
    sxx.index.name = 'frequency'
    sxx.columns.name = 'Time'
   
    if show or save:
        # plotting the spectrogram
        fig, ax = plt.subplots(figsize = figsize)
        fig_title = 'Spectrogram of %s'%ts_sr.name if len(ts_sr.name) > 0 else 'Spectrogram'
        fig.suptitle(fig_title, y = 1.04, fontsize = 15, fontweight = 'bold')
        heatdata = sxx.loc[sxx.index[int(vis_frac[0] * sxx.shape[0])] : sxx.index[int(vis_frac[1] * sxx.shape[0]) - 1]].sort_index(ascending = False)
        sns.heatmap(data = heatdata, cbar = True, ax = ax)
        fig.tight_layout()
        if show: plt.show()
        if save and savepath is not None:
            os.makedirs(savepath, exist_ok=True)
            fname = fig_title if fname is None else fname
            fig.savefig('%s/%s.jpg'%(savepath, fname.replace(' ', '_')), bbox_inches = 'tight')
        plt.close()
   
    # returning the spectrogram
    if ret_sxx: return sxx


def invalid_bins(msg):
    raise InvalidBins(msg)
def grouped_mode(data, bins = None, neglect_values=[], neglect_above=None, neglect_bellow=None, neglect_quan=0):
    """
    The purpose of this function is to calculate mode (mode of mean-median-mode) value

    Inputs:
        :data: pandas Series, list, numpy ndarray - must be 1-D
        :bins: int, list or ndarray, indicates bins to be tried to calculate mode value. Default is None
        :neglect_values: list, ndarray, the values inside the list will be removed from data. Default is []
        :neglect_above: float, values above this will be removed from the data. Default is None
        :neglect_beloow: float, values bellow this will be removed from the data. Default is None
        :neglect_quan: 0 < float < 1 , percentile range which will be removed from both sides from data distribution. Default is 0

    Outputs:
        mode for the time series data
    """

    if not isinstance(data, pd.Series): data = pd.Series(data)
    if neglect_above is not None: data = data[data <= neglect_above]
    if neglect_bellow is not None: data = data[data >= neglect_bellow]
    if len(neglect_values) > 0: data.replace(to_replace = neglect_values, value = np.nan, inplace = True)
    data.dropna(inplace = True)
    if neglect_quan != 0: data = data[(data >= data.quantile(q = neglect_quan)) & (data <= data.quantile(q = 1 - neglect_quan))]
    if isinstance(bins, type(None)): bins = np.arange(10, 101, 10)
    elif isinstance(bins, int): bins = [bins]
    elif isinstance(bins, list) or isinstance(bins, np.ndarray): pass
    else: invalid_bins('Provided bins are not valid! Please check requirement for bins.')
    _mode = []
    for _bins in bins:
        mode_group = pd.cut(data, bins = _bins).value_counts()
        max_val = mode_group.values[0]
        if max_val == 1: break
        mode_group = mode_group[mode_group == max_val]
        _mode.append(np.mean([i.mid for i in mode_group.index]))
    return np.mean(_mode)


def get_category_edges(dt, categories=None, names=None):
    """
    This function calculates edges for categorical values and returns start, stop, duration and interval
    
    Inputs:
        :dt: pandas Series, numpy array or list, time series signal containing categorical variable values.
        :categories: list or None. all possible values for the signal dt. If None, then unique values found in dt \n
                     are considered only possible values for dt.
        :names: dict, contains name of span DataFrame for each unique value of dt. 
                Default is None means name of span DataFrame will be designated by its value
    
    Outputs:
        :spans: list of pandas DataFrame, each DataFrame contains four columns 'start', 'stop', 'duration', 'interval' \n
                which describes the edges of each categorical value in categories.
    """
    
    if isinstance(dt, (list, np.ndarray)):
        dt = pd.Series(dt)
    elif isinstance(dt, pd.DataFrame):
        dt = dt.iloc[:, 0]
    elif isinstance(dt, pd.Series):
        pass
    else:
        raise ValueError('type %s for dt isnt supported (can be list, pandas Series, numpy array)'%type(dt))
    
    if categories is None:
        categories = np.sort(dt.unique())
    
    spans = {cat: [] for cat in categories}
    st = dt.index[0]
    cat = dt.iloc[0]
    for i in range(1, dt.shape[0]):
        if cat != dt.iloc[i]:
            spans[cat].append([st, dt.index[i-1]])
            st = dt.index[i]
            cat = dt.iloc[i]
    spans[dt.values[-1]].append([st, dt.index[-1]])
    
    for cat in categories:
        spans[cat] = pd.DataFrame(spans[cat], columns=['start', 'stop'])
        if names is None:
            names = {}
        names[cat] = str(cat)
        spans[cat].columns.name = names[cat]
    spans = list(spans.values())
    
    return spans


def get_edges_from_ts(sr, th_method='median', th_factor=.5, th=None,  del_side='up', name=None):
    """
    The purpose of this function is to find the edges specifically indices of start and end 
    of certain regions by thresholding the desired time series data.

    Inputs:
        :sr: pandas Series, time series data with proper timestamp as indices of the series. Index timestamps must be timedelta type values.
             Index timestamps must be sorted from old to new time (ascending order).
        :th_method: {'mode', 'median', 'mean'}, method of calculating threshold, Default is 'median'
        :th_factor: flaot/int, multiplication factor to be multiplied with th_method value to calculate threshold. Default is .5
        :th: int/flaot, threshold value inserted as argument. Default is None
        :del_side: {'up', 'down'}, indicating which side to be removed to get edges. Default is 'up'
        :name: name of the events, default is None

        Note: the algorithm starts recording when it exceeds the threshold value, so open interval system.
    
    Outputs:
        :edges: pandas DataFrame, contains columns 'start', 'end', 'duration', 'interval' etc. 
                related to the specific events found after thresholding
    """

    if th is None:
        if th_method == 'median': th = sr.median() * th_factor
        elif th_method == 'mean': th = sr.mean() * th_factor
    state = False    # keeps the present state, True if its within desired edges
    edges = [[], []]    # keeps the edges, start and stop
    if del_side == 'up':
        for i in range(sr.shape[0]):
            if sr.iloc[i] < th and not state:
                state = True
                edges[0].append(sr.index[i])
            elif sr.iloc[i] >= th and state:
                state = False
                edges[1].append(sr.index[i - 1])
    elif del_side == 'down':
        for i in range(sr.shape[0]):
            if sr.iloc[i] > th and not state:
                state = True
                edges[0].append(sr.index[i])
            elif sr.iloc[i] <= th and state:
                state = False
                edges[1].append(sr.index[i - 1])
    if state:
        edges[1].append(sr.index[-1])
    edges = pd.DataFrame(edges, index = ['start', 'stop']).T.reset_index(drop = True)
    edges['duration'] = edges['stop'] - edges['start']
    edges['interval'] = np.nan
    edges['interval'].iloc[1:] = [j - i for i,j in zip(edges['stop'].iloc[:-1], edges['start'].iloc[1:])]
    if name is None:
        if sr.name is not None:
            edges.columns.name = '%s_edges'%sr.name
        else:
            edges.columns.name = 'edges'
    else:
        edges.columns.name = name
    return edges


def each_row_max(data):
    """
    The purpose of this function is to get the maximum values and corresponding column names for each row of a matrix

    Inputs:
        :data: list of lists/numpy ndarray or pandas dataframe, matrix data from where max values will be calculated
    
    Outputs:
        returns same data with two new columns with max values and corresponding column names
    """

    if isinstance(data, np.ndarray) or isinstance(data, list): data = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame): pass
    else: InputVariableError('input data type must be list of lists or pandas dataframe or numpy ndarray!')
    if data.isnull().all().all():
        data['max_val'] = np.nan
        data['max_col'] = np.nan
    else:
        col = data.columns
        row = data.index
        data = data.values
        max_idx = np.nanargmax(data, axis=1)
        max_val = [data[i, max_idx[i]] for i in range(data.shape[0])]
        max_col = col[max_idx]
        data = pd.DataFrame(data, index = row, columns = col)
        data['max_val'] = max_val
        data['max_col'] = max_col
    return data


def moving_slope(df, fs=None, win=60, take_abs=False, nan_valid=True):
    """
    The purpose of this function is to calculate the slope inside a window for a variable in a rolling method
   
    Inputs
        :df: pandas DataFrame or Series, contains time series columns
        :fs: str, the base sampling period of the time series, default is the mode value of the difference in time between two consecutive samples
        :win: int, window length, deafult is 60
        :take_abs: bool, indicates whether to take the absolute value of the slope or not. Default is False\n
                   returns the pandas DataFrame containing resultant windowed slope values. Default is True
    
    Outputs:
        :new_df: pandas DataFrame, new dataframe with calculated rolling slope
    """
    
    # checking and resampling by the sampling frequency
    isdf = True
    if isinstance(df, pd.Series):
        isdf = False
        df = df.to_frame()
    df = df.resample(rule = fs, closed = 'right', label = 'right').last()
   
    # creating index matrix for the window
    x = np.arange(1, win + 1)[:, np.newaxis].astype(float)
    xmat = np.tile(x, (1, df.shape[1]))
   
    # loop initialization
    dfidx = df.index
    new_df = pd.DataFrame(np.ones(df.shape) * np.nan, columns  = ['%s_slope'%i for i in df.columns])
    perc1 = int(df.shape[0] * .01)
    cnt = perc1
    print('\r  0 percent completed...', end = '')
    for i in range(win, df.shape[0] + 1):
        if i == cnt:
            cnt += perc1
            print('\r%3d percent completed...'%(cnt // perc1), end = '')
        y = df.iloc[i - win : i, :].values
        if nan_valid: xy_bar = np.nanmean(np.multiply(xmat, y), axis = 0)
        else: xy_bar = np.mean(np.multiply(xmat, y), axis = 0)
        xnan = xmat.copy()
        xnan[np.isnan(y)] = np.nan
        if nan_valid: x_bar = np.nanmean(xnan, axis = 0)
        else: x_bar = np.mean(xnan, axis = 0)
        if sum(~np.isnan(x_bar)) != 0:
            if nan_valid: xsq_bar = np.nanmean(xnan ** 2, axis = 0)
            else: xsq_bar = np.mean(xnan ** 2, axis = 0)
            x_barsq = x_bar ** 2
            # calculating slope
            den = x_barsq - xsq_bar
            den[den == 0] = np.nan
            if nan_valid: m = (x_bar * np.nanmean(y, axis = 0) - xy_bar) / den
            else: m = (x_bar * np.nanmean(y, axis = 0) - xy_bar) / den
            new_df.loc[i - 1] = abs(m) if take_abs else m
    print('\r100 percent completed... !!')
    new_df.index = dfidx
    if not isdf: new_df = new_df.iloc[:, 0]
    return new_df


def standardize(data, zero_std = 1):
    """
    This function applies z-standardization on the data

    Inputs:
        :data: pandas series, dataframe, list or numpy ndarray, input data to be standardized
        :zero_std: float, value used to replace std values in case std is 0 for any column. Default is 1
    
    Outputs:
        standardized data
    """

    dtype = 0
    if isinstance(data, pd.Series):
        data = data.to_Frame()
        dtype = 1
    elif isinstance(data, list) or isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
        dtype = 2
    elif isinstance(data, pd.DataFrame): pass
    else: raise ValueError('Provided data is inappropriate! ')
    data_std = data.std()
    data_std[data_std == 0] = zero_std
    data = (data - data.mean()) / data_std
    if dtype == 0: return data
    elif dtype == 1: return data[data.columns[0]]
    else : return data.values


def normalize(data, zero_range = 1, method = 'min_max_0_1'):
    """
    The purpose of this function is to apply normalization of the input data

    Inputs:
        :data: pandas series, dataframe, list or numpy ndarray, input data to be standardized
        :zero_range: float, value used to replace range values in case range is 0 for any column. Default is 1
        :method: {'zero_mean', 'min_max_0_1', 'min_max_-1_1'}.\n
            'zero_mean' : normalizes the data in a way that makes the data mean as 0\n
            'min_max_0_1' : normalizes the data in a way that the data becomes confined within 0 and 1\n
            'min_max_-1_1' : normalizes the data in a way that the data becomes confined within -1 and 1\n
            Default is "min_max_0_1"\n
    
    Outputs:
        Normalized data
    """

    dtype = 0
    if isinstance(data, pd.Series):
        data = data.to_Frame()
        dtype = 1
    elif isinstance(data, list) or isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
        dtype = 2
    elif isinstance(data, pd.DataFrame): pass
    else: raise ValueError('Provided data is inappropriate! ')
    data_range = data.max() - data.min()
    data_range[data_range == 0] = zero_range
    if method == 'min_max_0_1': data = (data - data.min()) / data_range
    elif method == 'min_max_-1_1': data = (data - data.min()) / data_range * 2 - 1
    elif method == 'zero_mean': data = (data - data.mean()) / data_range
    if dtype == 0: return data
    elif dtype == 1: return data[data.columns[0]]
    else : return data.values


def feature_evaluator(data, label_name, label_type_num, n_bin=40, is_all_num=False, is_all_cat=False, num_cols=[], 
                      cat_cols=[], cmap='gist_rainbow', fig_width=30, fig_height=5, rotation=40, show=True, save=False, 
                      savepath=None, fname=None, fig_title=''):
    """
    This function calculates feature importance by statistical measures. 
    Here its okay to use labels before converting into one hot encoding, its even better not to convert into one hot.
    
    The process is below-

    --------- feature evaluation criteria --------
    
    for numerical feature evaluation:\n
    group- a variable is divided into groups based on a constant interval of its values\n
    group diversity- std of mean of every group\n
    group uncertainty- mean of std of points for every group\n
    
    for categorical feature evaluation:\n
    group diversity- mean of std of percentage of categories inside a feature\n
    group uncertainty- mean of std of percentage of groups for each category\n

    group confidence (Importance)- group diversity / group uncertainty

    Inputs:
        :data: must be a pandas dataframe containing feature and label data inside it\n
              (it must contain only features and labels, no unnecessary columns are allowed)
        :label_name: list; containing names of the columns used as labels
        :label_type_num: list of bool, containing flag info if the labels are numerical or not (must be corresponding to label_name)
        :n_bin: int, expressing number of divisions of the label values.\n
               If any label is categorical, n_bin for that label will be equal to the number of categories.\n
               Default is 40 for numerical labels.
        :is_all_num: bool, is a simplification flag indicates whether all the features are numerical or not
        :is_all_cat: bool, is a simplification flag indicates whether all the features are categorical or not
        :num_cols: list, containing name of numerical columns if needed
        :cat_cols: list, containing categorical feature names if needed
        :cmap: str, is the matplotlib colormap name, default is 'gist_rainbow'
        :fig_width: float, is the width of figure, default is 30
        :fig_height: float, is the height of figure, default is 5
        :rotation: float, rotational angle of x axis labels of the figure, default is 40
        :show: bool, whether to show the graph or not, default is True
        :save: bool, whether to save the figure or not, default is False
        :savepath: str, path where the data will be saved, default is None
        :fname: str, filename which will be used for saving the figure, default is None
        :fig_title: str, title of the figure, default is 'Feature Confidence Evaluation for categorical Features'
        
    Outputs:
        :num_lab_confids: dict, contains confidence values for each label
    """    
    
    # separating numerical and categorical feature data 
    if is_all_num:
        num_cols = data.drop(label_name, axis = 1).columns.to_list()
    elif is_all_cat:
        cat_cols = data.drop(label_name, axis = 1).columns.to_list()
    else:
        if len(cat_cols) > 0:
            num_cols = list(data.drop(list(cat_cols) + label_name, axis = 1).columns)
        elif len(num_cols) > 0:
            cat_cols = list(data.drop(list(num_cols) + label_name, axis = 1).columns)
    num_data = data[num_cols].copy()
    cat_data = data[cat_cols].copy()
    # numerical feature normalization (min_max_normalization)
    num_data = (num_data - num_data.min()) / (num_data.max() - num_data.min())
    # taking in numerical and categorical labels
    label_name = np.array(label_name)
    num_labels = data[label_name[label_type_num]].copy()
    cat_labels = data[label_name[~np.array(label_type_num)]]
    
    # n_bin arrangements
    default_n_bin = 25
    if n_bin is None:
        n_bin = [ default_n_bin for i in label_type_num]
    else:
        n_bin = [ n_bin for i in label_type_num]
    num_bin = np.array(n_bin)[label_type_num]
    cat_bin = [cat_labels[i].unique().shape[0] for i in cat_labels.columns]
    
    # loop definitions
    labels_list = [num_labels, cat_labels]
    labels_tag = ['Numerical label', 'Categorical label']
    nbins = [num_bin, cat_bin]
    fig_title = 'Feature Confidence Evaluation' if fig_title == '' else fig_title
    # to collect all values of evaluation
    num_lab_confids = {}
    # loop for numerical and categorical labels
    for l, (labels, nbin) in enumerate(zip(labels_list, nbins)):
        num_lab_confids[labels_tag[l]] = {}
        # plot definitions
        nrows = labels.columns.shape[0]
        if not is_all_cat and nrows > 0:
            fig_num, ax_num = plt.subplots(figsize = (fig_width, fig_height * nrows), nrows = nrows)
            fig_tnum = fig_title + ' for Numerical Features'
            fig_num.suptitle(fig_tnum, y = 1.06, fontsize = 20, fontweight = 'bold')
        if not is_all_num  and nrows > 0:
            fig_cat, ax_cat = plt.subplots(figsize = (fig_width, fig_height * nrows), nrows = nrows)
            fig_tcat = fig_title + ' for Categorical Features'
            fig_cat.suptitle(fig_tcat, y = 1.06, fontsize = 20, fontweight = 'bold')
        # special case for nrows = 1
        if nrows == 1:
            if not is_all_cat:
                ax_num = [ax_num]
            if not is_all_num:
                ax_cat = [ax_cat]
        # loops for labels
        for i, label in enumerate(labels.columns):
            print('Evaluation for label : %s (%s) ....'%(label, labels_tag[l]), end = '')
            # dividing the data set into classes depending on the label/target values
            divs = pd.cut(labels[label], bins = nbin[i])
            divs = divs.replace(to_replace = divs.unique().categories, value = np.arange(divs.unique().categories.shape[0]))
            
            if not is_all_cat:
                # ------------- calculation for numerical data ----------------------
                # grouping features depending on the segments of the label
                num_data['%s_segments'%label] = divs
                group = num_data.groupby('%s_segments'%label)
                # calculation of confidence of features and sorting in descending order
                grp_confid = group.mean().std() / group.std().mean()
                grp_confid = grp_confid.sort_values(ascending = False).to_frame().T
                num_lab_confids[labels_tag[l]]['%s_numerics'%label] = grp_confid
                # removing segment column 
                num_data.drop(['%s_segments'%label], axis = 1, inplace = True)
                # plotting 
                sns.barplot(data = grp_confid, ax = ax_num[i], palette = cmap)
                ax_num[i].set_xticklabels(labels = grp_confid.columns, rotation = rotation, ha = 'right')
                ax_num[i].set_xlabel('Features', fontsize = 15)
                ax_num[i].set_ylabel('Confidence Value', fontsize = 15)
                ax_num[i].grid(axis = 'y', color = 'white', alpha = 1)
                ax_num[i].set_title('Feature Confidence for label: %s'%label, fontsize = 15)
            
            if not is_all_num:
                # ------------- calculation for categorical data ----------------------
                # adding label and category of label columns in the categorical data set
                cat_data[label] = labels[label].copy()
                cat_data['%s_segments'%label] = divs
                # calculation of mean of std of percentage 
                cat_confid = pd.Series(index = cat_data.columns[:-2], dtype=np.float64)
                for j, col in enumerate(cat_data.columns[:-2]):
                    temp = cat_data.groupby(['%s_segments'%label, col])[label].count().unstack(level = 0).fillna(0)
                    temp = temp / temp.sum()
                    cat_confid.loc[col] = temp.std().mean() * temp.T.std().mean()
                # sorting confidence values according to descending order 
                cat_confid = cat_confid.sort_values(ascending = False).to_frame().T
                num_lab_confids[labels_tag[l]]['%s_categs'%label] = cat_confid
                # removing the label columns
                cat_data.drop(['%s_segments'%label, label], axis = 1, inplace = True)
                # plotting 
                sns.barplot(data = cat_confid, ax = ax_cat[i], palette = cmap)
                ax_cat[i].set_xticklabels(labels = cat_confid.columns, rotation = rotation, ha = 'right')
                ax_cat[i].set_xlabel('Features', fontsize = 15)
                ax_cat[i].set_ylabel('Confidence Value', fontsize = 15)
                ax_cat[i].grid(axis = 'y', color = 'white', alpha = 1)
                ax_cat[i].set_title('Feature Confidence for label: %s'%label, fontsize = 15)
            print('  done')
        
        if not is_all_cat and nrows > 0:
            fig_num.tight_layout()
            if save and savepath is not None:
                if fname is None: fname = fig_tnum
                if savepath[-1] != '/': savepath += '/'
                fig_num.savefig('%s%s.jpg'%(savepath, fname), bbox_inches = 'tight')
        if not is_all_num and nrows > 0:
            fig_cat.tight_layout()
            if save and savepath is not None:
                if fname is None: fname = fig_tcat
                if savepath[-1] != '/': savepath += '/'
                fig_cat.savefig('%s%s.jpg'%(savepath, fname), bbox_inches = 'tight')
        if show: plt.show()
        if nrows > 0: plt.close()
    return num_lab_confids


def get_weighted_scores(score, y_test):
    """
    This function calculates weighted average of score values we get from class_result function

    Inputs:
        :score: pandas DataFrame, contains classifiaction scores that we get from msd.class_result() function
        :y_test: numpy array, pandas Series or python list, contains true classification labels

    Outputs:
        :_score: pandas DataFrame, output score DataFrame with an additional column containing weighted score values
    """
    _score = score.copy()
    score_cols = _score.columns.drop('average').to_list()
    counts = pd.Series(y_test).value_counts()
    countidx2str = {i: str(i) for i in counts.index}
    countstr2idx = {countidx2str[i]: i for i in countidx2str}
    counts = counts.to_frame().T

    for col in score_cols:
        if col not in countstr2idx:
            counts[col] = 0
        else:
            counts[col] = counts[countstr2idx[col]]
            counts.drop([countstr2idx[col]], axis=1, inplace=True)
    _score['weighted_average'] = (_score.drop('accuracy').drop('average', axis=1)[score_cols].values * counts[score_cols].values / counts.values.sum()).sum(axis=1).tolist()+[_score['average'].loc['accuracy']]
    return _score


def class_result(y, pred, out_confus=False):
    """
    This function computes classification result with confusion matrix
    For classification result, it calculates precision, recall, f1_score, accuracy and specificity for each classes.

    Inputs:
        :y: numpy 1-d array, list or pandas Series, true classification labels, a vector holding class names/indices for each sample
        :pred: numpy 1-d array, list or pandas Series, predicted values, a vector containing prediction opt for class indices similar to y.\n
               Must be same length as y
        :out_confus: bool, returns confusion matrix if set True. Default is False
    
    Outputs:
        :result: DataFrame containing precision, recall and f1 scores for all labels
        :con_mat: DataFrame containing confusion matrix, depends on out_confus
    """
    
    if not any([isinstance(y, np.ndarray), isinstance(y, pd.Series)]):
        y = np.array(y)
    if not any([isinstance(pred, np.ndarray), isinstance(pred, pd.Series)]):
        pred = np.array(pred)
        
    y = y.ravel()
    pred = pred.ravel()
    y_labels = set(np.unique(y))
    p_labels = set(np.unique(pred))
    labels = sorted(list(y_labels | p_labels))
    con_mat = pd.DataFrame(np.zeros((len(labels), len(labels))).astype(int),
                           columns = labels, index = labels)    #row--> true, col--> pred
    con_mat.index.name = 'true'
    con_mat.columns.name = 'prediction'
    for lab_true in labels:
        vy = set(np.where(y == lab_true)[0])
        for lab_pred in labels:
            vpred = set(np.where(pred == lab_pred)[0])
            con_mat[lab_pred].loc[lab_true] = len(vy & vpred)
    prec = []
    recall = []
    f_msr = []
    acc = []
    spec = []
    sum_pred = con_mat.sum()
    sum_true = con_mat.T.sum()
    total = y.shape[0]
    for label in labels:
        tn = total - (sum_true.loc[label] + sum_pred.loc[label] - con_mat[label].loc[label])
        fp = sum_pred.loc[label] - con_mat[label].loc[label]
        if (tn+fp) != 0:
            sp = tn / (tn + fp)
        else:
            sp = 0
        spec.append(sp)
        if sum_pred.loc[label] != 0:
            pr = con_mat[label].loc[label] / sum_pred.loc[label]
        else:
            pr = 0
        prec.append(pr)
        if sum_true.loc[label] != 0:
            rec = con_mat[label].loc[label] / sum_true.loc[label]
        else:
            rec = 0
        recall.append(rec)
        if pr + rec != 0:
            f = 2 * pr * rec / (pr + rec)
        else:
            f = 0
        f_msr.append(f)
        acc.append(np.nan)
    result = pd.DataFrame([spec, prec, recall, f_msr], columns = labels, index = ['specificity', 'precision', 'recall', 'f1_score'])
    result.columns = [str(c) for c in result.columns]
    avg = result.T.mean()
    avg.name = 'average'
    result = pd.concat((result, avg), axis = 1, sort = False)
    result.columns.name = 'labels'
    acc = pd.DataFrame([acc + [np.trace(con_mat.values) / y.shape[0]]], columns = result.columns, index = ['accuracy'])
    result = pd.concat((result, acc), axis = 0, sort = False)
    result = get_weighted_scores(result, y)
    if out_confus:
        return result, con_mat
    else:
        return result


def rsquare_rmse(y, pred):
    """
    This function calculates R-square value (coefficient of determination) and root mean squared error value for regression evaluation.

    Inputs:
        :y: numpy 1-d array, list or pandas Series, contains true regression labels
        :pred: numpy 1-d array, list or pandas Series, prediction values against y. Must be same size as y.
    
    Outputs:
        :r_sq: calculated R-square value (coefficient of determination)
        :rmse: root mean square error value
    """

    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    y = np.squeeze(y)
    pred = np.squeeze(pred)

    y_mean = np.mean(y)
    r_sq = 1 - np.sum((pred - y) ** 2) / np.sum((y - y_mean) ** 2)
    rmse = np.sqrt(np.mean((y - pred) ** 2))
    return r_sq, rmse


def regression_result(y, pred):
    """
    This function calculates R-square value (coefficient of determination) and root mean squared error value for regression evaluation.

    Inputs:
        :y: numpy 1-d array, list or pandas Series, contains true regression labels
        :pred: numpy 1-d array, list or pandas Series, prediction values against y. Must be same size as y.
    
    Outputs:
        :r_sq: float, calculated R-square value (coefficient of determination)
        :rmse: float, root mean square error value
        :corr: float, correlation coefficient between y and pred
    """

    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    y = np.squeeze(y)
    pred = np.squeeze(pred)

    rsquare, rmse = rsquare_rmse(y, pred)
    corr = np.corrcoef(y, pred)[1, 0]

    return rsquare, rmse, corr


def one_hot_encoding(arr, class_label=None, ret_label=False, out_type='ndarray'):
    """
    This function converts categorical values into one hot encoded format.

    Inputs:
        :arr: numpy 1-d array, list or pandas Series, containing all class names/indices
        :class_label: numpy array or list, list or pandas Series, containing only class labels inside 'arr'
        :out_type: {'ndarray', 'dataframe', 'list'}, data type of the output
    
    Outputs:
        :label: one hot encoded output data, can be ndarray, list or pandas DataFrame based on 'out_type'
        :class_label: names of class labels corresponding to each column of one hot encoded data.\n
                      It is provided only if ret_label is set as True.
    """
    if isinstance(arr, pd.Series): pass
    else: arr = pd.Series(arr)
    if class_label is None: class_label = np.sort(arr.unique())
    elif isinstance(class_label, list): class_label = np.array(class_label)
    label = pd.DataFrame(np.zeros((arr.shape[0], class_label.shape[0])), index = arr.index, columns = class_label)
    for lb in class_label:
        label[lb].loc[arr[arr == lb].index] = 1
    label = label.loc[~np.all(label.values == 0, axis = 1)]
    if out_type == 'ndarray': label = label.values
    elif out_type == 'list': label = [label.loc[i].to_list() for i in label.index]
    if ret_label: return label, class_label
    else: return label


class SplitDataset():
    """
    This class contains method to split data set into train, validation and test.
    It allows both k-fold cross validation and casual random splitting.
    Additionally it allows sequence splitting, necessary to model time series data using LSTM.
    
    Inputs:
        :data: pandas DataFrame or Series or numpy ndarray of dimension #samples X other dimensions with rank 2 or higher in total
        :label: pandas dataframe or series or numpy ndarray with dimension #samples X #classes with rank 2 in total
        :index: numpy array, we can explicitly insert indices of the data set and labels, default is None
        :test_ratio: float, ratio of test data, default is 0
        :np_seed: int, numpy seed used for random shuffle, default is 1216
        :same_ratio: bool, keep the test and validation ratio same for all classes.\n
                        Should be only true if the labels are classification labels, default is False
        :make_onehot: bool, if True, the labels will be converted into one hot encoded format. Default is False
    """
    
    def __init__(self, data, label, index=None, test_ratio=0, np_seed=1216, same_ratio=False, make_onehot=False):
        
        # checking initial criteria and fixing stuffs
        if index is None:
            if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
                index = data.index
            elif isinstance(label, pd.DataFrame) or isinstance(label, pd.Series):
                index = label.index
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series): data = data.values
        if isinstance(label, pd.DataFrame) or isinstance(label, pd.Series): label = label.values
        if isinstance(data, list): data = np.array(data)
        if isinstance(label, list): label = np.array(label)
        
        # defining data and label parameter
        self.data = data.copy()
        self.label = one_hot_encoding(label) if make_onehot else label.copy()
        self.data_shape = self.data.shape
        self.test_ratio = test_ratio
        # checking mismatch in data and label sizes
        self.check_data_label()
        # setting balanced sampling for different class labels if same_ratio = True
        self.prepare_class_index(same_ratio)
        self.index = np.arange(self.data_shape[0]) if index is None else index
        # shuffling indices
        for k in self.idx:
            # setting numpy random seed
            np.random.seed(np_seed)
            np.random.shuffle(self.idx[k])
        # setting test length based on
        self.test_len = {i : int(self.idx[i].shape[0] * self.test_ratio) for i in self.idx}
        # initializing the input and output data
        self.dataset = {'data' : self.data.copy(), 'label' : self.label.copy(), 'index' : self.index.copy()}
        del self.data, self.label, self.index
       
    # raises error is data and label shapes do not match
    def data_label_mismatch(self, msg):
        raise DataLabelMismatchError(msg)
    def split_method_error(self, msg):
        raise SplitMethodError(msg)
   
    # checks whether data and label shapes match properly or not by different criteria
    def check_data_label(self):
        if self.data_shape[0] != self.label.shape[0]:
            self.data_label_mismatch('data shape is %s whence label shape is %s (first dimensions of both data and label must be same.))'%(self.data_shape, self.label.shape))
        elif self.data_shape[0] == 0 or self.label.shape[0] == 0:
            self.data_label_mismatch('data_shape : %s, label_shape : %s\nfirst dimension should not be zero!'%(self.data_shape, self.label.shape))
        else:
            if len(self.data_shape) == 1:
                self.data = self.data[:, np.newaxis]
                self.data_shape = self.data.shape
            if len(self.label.shape) == 1: self.label = self.label[:, np.newaxis]
            elif len(self.label.shape) > 2: self.data_label_mismatch('label rank is more than 2 !!')
    
    # gets the indices depending on class labels for classification type true label
    def prepare_class_index(self, same_ratio):
        if self.label.shape[1] == 1:
            if same_ratio:
                self.cls = np.sort(np.unique(self.label.ravel()))
                self.idx = {i : np.where(self.label.ravel() == i)[0] for i in self.cls}
            else:
                self.cls = np.array([0])
                self.idx = {0 : np.arange(self.data_shape[0])}
        else:
            self.cls = np.arange(self.label.shape[1])
            self.idx = {i : np.where(self.label[:, i] == 1)[0] for i in self.cls}
    
    def random_split(self, val_ratio = .15):
        """
        This method splits the data randomly into train, validation and test sets.

        Inputs:
            :val_ratio: float, validation data set ratio, default is .15
            
        Outputs:
            :outdata: dict containing data, label and indices for train, validation and test data sets
        """
        self.val_len = {i : int(self.idx[i].shape[0] * val_ratio) for i in self.idx}
        # defining starting and ending indices
        sted = {'test' : {'st': {i : 0 for i in self.cls}, 'ed' : self.test_len},
                'validation' : {'st' : self.test_len, 'ed' : {i : self.test_len[i] + self.val_len[i] for i in self.cls}},
                'train' : {'st' : {i : self.test_len[i] + self.val_len[i] for i in self.cls},
                         'ed' : {i : self.data_shape[0] for i in self.cls}}}
        # defining final data and storing data inside it
        outdata = {_set : {lb : [] for lb in ['data', 'label', 'index']} for _set in ['train', 'validation', 'test']}
        for _set in outdata:
            for val in outdata[_set]:
                for cl in self.cls:
                    outdata[_set][val].append(self.dataset[val][self.idx[cl][sted[_set]['st'][cl] : sted[_set]['ed'][cl]]])
                outdata[_set][val] = np.concatenate(outdata[_set][val], axis = 0)
        return outdata
    
    # ###############  CROSS_VALIDATION_SPLIT  ####################
    # this function is prepared to run cross validation using proper data and labels

    def cross_validation_split(self, fold=5, only_index=False):
        """
        This method applies cross validation splitting.

        Inputs:
            :fold: int, number of folds being applied to the data, default is 5
            :only_index: bool, whether to return only index or data, label, index all.
                         Default is False meaning the function will return data, label and index for all folds.
        
        Outputs:
            :outdata: dict containing data, label and indices for train, validation and test data sets
        """
        
        #defining validation length for each type of class labels
        self.val_len = {i : (self.idx[i].shape[0] - self.test_len[i]) // fold for i in self.cls}
        # defining starting and ending indices
        sted = {'test' : {'st': {i : 0 for i in self.cls}, 'ed' : self.test_len},
                'validation' : {'st' : {'fold_%d'%i : {j : self.test_len[j] + (i - 1) * self.val_len[j] for j in self.cls} for i in range(1, fold + 1)},
                                'ed' : {'fold_%d'%i : {j : self.test_len[j] + i * self.val_len[j] for j in self.cls} for i in range(1, fold + 1)}},
                'train' : {'st' : {'fold_%d'%i : {j : [self.test_len[j], self.test_len[j] + i * self.val_len[j]] for j in self.cls} for i in range(1, fold + 1)},
                           'ed' : {'fold_%d'%i : {j : [self.test_len[j] + (i - 1) * self.val_len[j], self.idx[j].shape[0]] for j in self.cls} for i in range(1, fold + 1)}}}
        outdata = {}
        for _set in ['train', 'validation', 'test']:
            outdata[_set] = {}
            for lb in ['data', 'label', 'index']:
                if (lb == 'index') or (lb in ['data', 'label'] and not only_index):
                    if _set == 'test':
                        outdata[_set][lb] = []
                        for j in self.cls:
                            outdata[_set][lb].append(self.dataset[lb][self.idx[j][sted[_set]['st'][j] : sted[_set]['ed'][j]]])
                        outdata[_set][lb] = np.concatenate(outdata[_set][lb], axis = 0)
                    else:
                        outdata[_set][lb] = {}
                        for i in range(1, fold + 1):
                            outdata[_set][lb]['fold_%d'%i] = []
                            for j in self.cls:
                                if _set == 'validation':
                                    outdata[_set][lb]['fold_%d'%i].append(self.dataset[lb][self.idx[j][sted[_set]['st']['fold_%d'%i][j] : sted[_set]['ed']['fold_%d'%i][j]]])
                                else:
                                    for k in range(2):
                                        outdata[_set][lb]['fold_%d'%i].append(self.dataset[lb][self.idx[j][sted[_set]['st']['fold_%d'%i][j][k] : sted[_set]['ed']['fold_%d'%i][j][k]]])
                            outdata[_set][lb]['fold_%d'%i] = np.concatenate(outdata[_set][lb]['fold_%d'%i], axis = 0)
        return outdata

    # returns data, label and indices for train, validation and test data sets as pydict
    def sequence_split(self, seq_len, val_ratio=.15, data_stride=1, label_shift=1,
                       split_method='multiple_train_val', sec=1, dist=0, inference=False):
        """
        This method creates sequences of time series data and then splits those sequence data into train, validation and test sets. 
        We can create mutiple pairs of train and validation data sets if we want.
        Test data set will be only one set.

        Note : As this method creates sequential data, it may also be needed in inference time.

        Inputs:
            :seq_len: int, length of each sequence
            :val_ratio: float, validation data set ratio, deafult is .15
            :data_stride: int, indicates the number of shift between two adjacent example data to prepare the data set, default is 1
            :label_shift: temporal difference between the label and the corresponding data's last time step, default is 1
            :split_method: {'train_val', 'multiple_train_val'}, default is 'multiple_train_val'
            :sec: int or pandas dataframe with columns naming 'start' and 'stop', number of sections for multiple_train_val method splitting (basically train_val splitting is multiple_train_val method with sec = 1), deafult is 1
            :dist: int, number of data to be leftover between two adjacent sec, default is 0
            :inference: bool, whether the method is called in inference or not. Default is False
        
        Outputs:
           :outdata: dict containing data, label and indices for train, validation and test data sets
        """
        
        # split_method
        if split_method == 'train_val':
            if sec == 1: sec = 1
            else: self.split_method_error("'train_val' type split method can not have sec = %s"%sec)
        test_fl = bool(self.test_ratio)    # test data flag
        skipsecs = sec * 2 - 1 + int(test_fl)    # number of segments to be considered for deleting data
        skiplen = seq_len + dist      # number of data indices to be removed
        tot_data = ((self.data_shape[0] - seq_len - label_shift - ((skiplen - data_stride) * skipsecs)) // data_stride) + 1    # total valid indices
        test_len = int(self.test_ratio * tot_data)    # total valid test indices
        data_per_sec = (tot_data - test_len) // sec    # total valid data for each section
        val_len_ps = int(val_ratio / (1 - self.test_ratio) * data_per_sec)      # number of validation indices per section
        tr_len_ps = int(data_per_sec - val_len_ps)       # number of train indices per section
        outdata = {x : {y : {'section_%d'%(i + 1) : [] for i in range(sec)} for y in self.dataset} for x in ['train', 'validation', 'test']}
        if inference:
            outdata = {'inference' : {m : [] for m in ['data', 'label', 'index']}}
            for st in range(self.data_shape[0] - seq_len + 1):
                outdata['inference']['data'].append(self.dataset['data'][st : st + seq_len])
                outdata['inference']['label'].append(self.dataset['label'][st + seq_len - 1 + label_shift])
                outdata['inference']['index'].append(self.dataset['index'][st + seq_len - 1])
            for k in outdata['inference']: outdata['inference'][k] = np.array(outdata['inference'][k])
            return outdata
        if test_fl:
            # test data starting indices
            sts = np.array([self.data_shape[0] - seq_len - label_shift - i * data_stride for i in range(test_len)])
            for st in sts:
                # storing data, label and index
                outdata['test']['data']['section_1'].append(self.dataset['data'][st : st + seq_len])
                outdata['test']['label']['section_1'].append(self.dataset['label'][st + seq_len - 1 + label_shift])
                outdata['test']['index']['section_1'].append(self.dataset['index'][st + seq_len - 1])
            for k in self.dataset: outdata['test'][k]['section_1'] = np.array(outdata['test'][k]['section_1'])
            st = self.data_shape[0] - seq_len - label_shift - (test_len - 1) * data_stride - skiplen
        else: st = self.data_shape[0] - seq_len - label_shift
        for i in range(sec, 0, -1):
            for j in range(val_len_ps):
                outdata['validation']['data']['section_%d'%i].append(self.dataset['data'][st : st + seq_len])
                outdata['validation']['label']['section_%d'%i].append(self.dataset['label'][st + seq_len - 1 + label_shift])
                outdata['validation']['index']['section_%d'%i].append(self.dataset['index'][st + seq_len - 1])
                st -= data_stride
            st -= (skiplen - data_stride)
            for j in range(tr_len_ps):
                outdata['train']['data']['section_%d'%i].append(self.dataset['data'][st : st + seq_len])
                outdata['train']['label']['section_%d'%i].append(self.dataset['label'][st + seq_len - 1 + label_shift])
                outdata['train']['index']['section_%d'%i].append(self.dataset['index'][st + seq_len - 1])
                st -= data_stride
            st -= (skiplen - data_stride)
        for nm in outdata:
            for k in outdata[nm]:
                for s in outdata[nm][k]:
                    outdata[nm][k][s] = np.flipud(np.array(outdata[nm][k][s]))
                if sec == 1 or nm == 'test':
                    outdata[nm][k] = outdata[nm][k]['section_1']
        return outdata


class paramOptimizer():
    """
    This class is used for hyper-parameter optimization for Machine Learning or other usage.

    Inputs:
        :params: dict of lists, containing choices for each of the param keys
        :mode: str, search mode. avaialble modes {'random', 'grid'}, default is 'random'
        :find: str, available {'min', 'max'}, indicates whether minimizing or maximizing the score, default is 'min'
        :iteration: int, number of iterations to be done, default is None
        :top: int, number of top scores to be stored; default is 3
        :rand_it_perc: float (0 ~ 1), indirectly sets the total number of iteration in 'random' model if iteration is not given, default is .5
        :shuffle_queue: bool, whether to shuffle the parameter queue or not, default is True
        :random_seed: float, random seed to use for random actions, default is 1216
    
    """
    def __init__(self, params, mode='random', find='min', iteration=None, top=3, rand_it_perc=.5, shuffle_queue=True, random_seed=1216):
        self.params = params
        self.paramlen = {p : len(params[p]) for p in params}
        self.queue = self.get_queue()
        self.total_it = len(self.queue)
        self.mode = mode
        if self.mode == 'random':
            if iteration is None: self.iteration = int(self.total_it * rand_it_perc)
            else: self.iteration = iteration
            shuffle_queue = True
            print('total number of iterations is : %d'%self.iteration)
        elif self.mode == 'grid': self.iteration = self.total_it
        if shuffle_queue:
            np.random.seed(random_seed)
            np.random.shuffle(self.queue)
        self.queue = pd.DataFrame(self.queue)
        self.queue['score'] = np.nan
        self.queue_cnt = 0
        self.top = top
        self.ascending = True if find == 'min' else False
        self.tops = pd.DataFrame(index = [i for i in range(self.top + 1)], columns = ['queue_idx', 'score'])
        self.storage = []
    
    def get_queue(self,):
        counts = [[p for p in self.params], [0 for _ in self.params], [len(self.params[p]) for p in self.params]]
        total_param = len(self.params)
        records = []
        while True:
            record = {p : self.params[p][i] for p, i in zip(counts[0], counts[1])}
            records.append(record)
            counts[1][0] += 1
            for pnum in range(1, total_param):
                if counts[1][pnum - 1] == counts[2][pnum - 1]: 
                    counts[1][pnum] += 1
                    counts[1][pnum - 1] = 0
            if counts[1][-1] == counts[2][-1] : break
        return records
    
    # gives the selected parameter
    def get_param(self,):
        return dict(self.queue.drop('score', axis = 1).iloc[self.queue_cnt])
    
    def set_score(self, score, element = ''):
        """
        This method takes in score value found for the current set of parameters and shows if the iteration is finished or not.

        Inputs:
            :score: float, the score value
            :element: anything, any additional storage element like model or any other variable etc.
        
        Outputs:
            :stop_iteration_flag: bool, if True, it means that the optimization is finished.
        """
        
        warnings.filterwarnings('ignore')
        self.queue['score'].iloc[self.queue_cnt] = score
        if self.queue_cnt < self.top + 1:
            self.tops.iloc[self.queue_cnt] = [self.queue_cnt, score]
            self.storage.append(element)
        else:
            self.tops.iloc[self.top] = [self.queue_cnt, score]
            self.storage[-1] = element
            self.tops.sort_values('score', ascending = self.ascending, inplace = True)
            self.storage = [self.storage[i] for i in self.tops.index]
            self.tops.reset_index(drop = True, inplace = True)
        self.queue_cnt += 1
        if self.queue_cnt == self.iteration: return True
        else: return False
    
    def best(self,):
        """
        this function provides the best parameter set in the end of optimization (can also be used at any time in stead of end)
        """

        return self.queue.iloc[self.tops['queue_idx'].values[:-1]].reset_index(drop = True), self.storage[:-1]

