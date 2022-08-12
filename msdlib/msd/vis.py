"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import matplotlib.lines as mlines
from matplotlib.cm import get_cmap
import seaborn as sns
import time
from scipy.stats import gaussian_kde
import datetime
from threading import Thread
from ..msdExceptions import WordLengthError, InputVariableError
import os


sns.set()
pd.plotting.register_matplotlib_converters()


# this is a custom designed progress bar for checking the loop timing. The user should follow this approach to make it work 
#with ProgressBar(arr, desc = 'intro of arr', perc = 5) as pbar:
#    for i in arr:
#        'your code/task inside the loop'
#        pbar.inc()


class ProgressBar():
    """
    Inputs:
        :arr: iterable, it is the array you will use to run the loop, it can be range(10) or any numpy array or python list or any other iterator
        :desc: str, description of the loop, default - 'progress'
        :barlen: int, length of the progress bar, default is 40
        :front space: int, allowed space for description, default is 20
        :tblink_max: float/int indicates maximum interval in seconds between two adjacent blinks, default is .4
        :tblink_min: float/int indicates minimum interval in seconds between two adjacent blinks, default is .18

    Outputs:
        there is no output. It creates a progress bar which shows the loop progress.
    """
    
    def __init__(self, arr, desc='progress', barlen=40, front_space = 20, tblink_max = .3, tblink_min = .18):
        self.xlen = len(arr)
        self.barlen = barlen
        if tblink_max >= 1:
            tblink_max = .9
            print("'tblink_max' was set to .9 seconds beacuse of exceeding maximum limit!")
        self.desc = desc[:front_space] + ' ' + '-' * (front_space - len(desc)) * int(front_space > len(desc))+ ' '
        self.barend = ' '*15
        self.tblmax = tblink_max
        self.tblmin = tblink_min
        self.blintv = self.tblmax    # blinking interval
        self.barincs = [int(self.xlen / self.barlen * (i + 1)) for i in range(self.barlen)]
        self.barinc = 0
        self.sym = '█'    # complete symbol in progress bar
        self.non = ' '    # gap symbol in progress bar
        self.blsyms = ['|', '/', '-', '\\']
        self.bllen = len(self.blsyms)
        self.blcnt = 0
        self.cnt = 0              # iterative counter for x elements
        self.barelap = 0          # iterative counter for progress bar
        self.set_barelap()        # setting proper next value for self.barinc
        self.blink = self.blsyms[0]
        self.tcntst = False
        self.min_tprint = .1      # minimum time interval for two consecutive bar print
        self.tprrec = time.time() - self.min_tprint - 1          # initialized with a bigger time 
    
    def __enter__(self, ):
        self.tst = time.time()
        # multithreading initialization
        self.thread = Thread(target = self.blink_func)
        self.flblink = True
        self.thread.start()
        # bar initialization
        self.prleftime = 'calculating..'
        self.tstack = 0
        self.tstackst = time.time()
        self.pastime = datetime.timedelta(seconds = 0)
        return self
    
    def calc_time(self):
        self.pastime = datetime.timedelta(seconds = time.time() - self.tst)
        self.leftime = self.pastime * (self.xlen / self.cnt - 1)
        self.tstackst = time.time()
        self.tstack = 0
        self.blintv = self.tblmax - (self.tblmax - self.tblmin) * (self.barelap + 1) / self.barlen
    
    def conv_time(self):
        d = self.pastime.days
        s = int(self.pastime.seconds + self.tstack)
        self.prpastime = '%s'%datetime.timedelta(days = d, seconds = s)
        if self.tcntst:
            d = self.leftime.days
            s = int(self.leftime.seconds - self.tstack)
            if d < 0:
                d, s = 0, 0
            self.prleftime = '%s'%datetime.timedelta(days = d, seconds = s)
    
    def set_barelap(self):
        if self.cnt == self.barincs[self.barinc]:
            if self.barinc < self.barlen - 1:
                self.barinc += 1
                while self.barincs[self.barinc] == self.barincs[self.barinc - 1] and self.barinc < self.barlen:
                    self.barinc += 1
            self.barelap = int(self.cnt / self.xlen * self.barlen)
    
    def inc(self):
        self.cnt += 1
        if not self.tcntst: self.tcntst = True
        self.set_barelap()
        self.calc_time()
        self.conv_time()
        self.barprint()
    
    def barprint(self, end = ''):
        if time.time() - self.tprrec >= self.min_tprint or not self.flblink:
            self.bar = self.sym * self.barelap + self.blink * int(self.flblink) + self.non * (self.barlen - self.barelap - int(self.flblink))
            self.pr = self.desc + '[' + self.bar + ']  ' + '%d/%d <%3d%%>'%(self.cnt, self.xlen, self.cnt / self.xlen * 100) + '  ( %s'%self.prpastime + ' < %s'%self.prleftime + ' )%s'%(self.barend)
            print('\r%s'%self.pr, end=end, flush=True)
            self.tprrec = time.time()
    
    def blink_func(self):
        while self.flblink:
            time.sleep(self.blintv)
            self.blcnt += 1
            if self.blcnt == self.bllen: self.blcnt = 0
            self.blink = self.blsyms[self.blcnt]
            # time adjustment part
            self.tstack = time.time() - self.tstackst
            self.conv_time()
            self.barprint()
    
    def __exit__(self, exception_type, exception_value, traceback):
        self.flblink = False
        time.sleep(self.tblmax)
        self.barend = ' Complete!' + ' '*15
        self.barprint('\n')


# the function bellow will divide a string into pieces based on a threshold of lenngth for each line, specially helps for labeling the plot axis names
def word_length_error(msg):
    raise msdExceptions.WordLengthError(msg)

# string : str, the input string
# maxlen : int, maximum allowable length for each line
def name_separation(string, maxlen):
    """
    This function separates a string based on its length and creates multiple lines and adds them finally to form new string with multiple lines

    Inputs:
        :string: str, input string
        :maxlen: int, maximum allowable length for each line
    
    Outputs:
        :newstr: str, output string after dividing the inserted string
    """
    
    strlen = len(string)
    if strlen > maxlen:
        words = string.split(' ')
        wlen = [len(i) for i in words]
        if max(wlen) > maxlen:
            word_length_error('One or more words have greater length than the maximum allowable length for each line !!')
        newstr = ''
        totlen = 0
        for i in range(len(wlen)):
            if totlen + wlen[i] + 1 > maxlen:
                totlen = wlen[i]
                newstr += '\n%s'%words[i]
            else:
                if i == 0:
                    totlen += wlen[i]
                    newstr += '%s'%words[i]
                else:
                    totlen += wlen[i] + 1
                    newstr += ' %s'%words[i]
    else: newstr = string
    return newstr


def input_variable_error(msg):
    raise msdExceptions.InputVariableError(msg)
def plot_time_series(same_srs, srs=[], segs=None, same_srs_width=[], spans=[], lines=[], linestyle=[], linewidth=[],
                     fig_title='', show=True, save=False, savepath=None, fname='', spine_dist=.035, spalpha=[],
                     ylims=[], name_thres=50, fig_x=30, fig_y=4, marker=None, xlabel='Time', ylabel='Data value', xrot=0, x_ha='center', axobj=None):
    """
    This function generates plots for time series data along with much additional information if provided as argument.
    This function provides many flexibilities such as dividing a time series into multiple subplots to visualize easily.
    It can plot multiple time series with proper legends, backgrounds and grids. It can also plot span plots, veritcal lines, each with separate legends.
    It allows multiple axes for multiple time series data with separate value ranges.

    Inputs:
        :same_srs: list of pandas series holding the variables which share same x-axis in matplotlib plot
        :srs: list of pandas series holding the variables which share different x axes, default is []
        :segs: int or pandas dataframe with two columns 'start' and 'stop' indicating the start and stop of each axis plot segment (subplot).\n
               Providing int will split the time series signals into that number of segments and will show as separate row subplot.\n
               Default is None
        :same_srs_width: list of flaots indicating each line width of corresponding same_srs time series data,\n
                        must be same size as length of same_srs, default is []
        :spans: list of pandas dataframe indicating the 'start' and 'stop' of the span plotting elements, default is []
        :lines: list of pandas series where the values will be datetime of each line position, keys will be just serials, default is []
        :linestyle: list of str, marker for each line, '' indicates continuous straight line, default is []
        :linewidth: list of constant, line width for each line, default is []
        :fig_title: title of the entire figure, default is ''
        :show: bool, indicating whether to show the figure or not, default is True
        :save: bool, indicating whether to save the figure or not, default is False
        :savepath: str, location of the directory where the figure will be saved, default is None
        :fname: str, figure name when the figure is saved, default is ''
        :spine_dist: constant indicating distance of the right side spines from one another if any, default is 0.035
        :spalpha: list of constants indicating alpha values for each of the span in spans, default is []
        :ylims: list of lists, indicating y axis limits in case we want to keep the limit same for all subplots, default is []
        :name_thres: int, maximum allowed characters in one line for plot labels, default is 50
        :fig_x: float, horizontal length of the figure, default is 30
        :fig_y: float, vertical length of each row of plot, default is 3
        :marker: str, marker for time series plots, default is None
        :xlabel: str, label name for x axis for each row, default is 'Time'
        :ylabel: str, label name for y axis for each row, default is 'Data value'
        :xrot: float, rotation angle of x-axis tick values, default is 0
        :x_ha: horizontal alignment of x axis tick values. It can be {'left', 'right', 'center'}. By default, it is 'center'.
        :axobj: matplotlib axes object, to draw time series on this axes object plot. Default is None.\n
                To use this option, segs must be 1 or None or dataframe with 1 row.
    
    Outputs:
        :axobj: Returns matplotlib axes object is axobj is provided in input.\n
                Otherwise, this function shows or stores plots, doesnt return anything
    """

    totsame = len(same_srs)
    totdif = len(srs)
    if totsame == 0:
        same_srs = [srs[0]]
        del srs[0]
        totsame = 1
        totdif -= 1
    if len(same_srs_width) == 0: same_srs_width = [.7 + i * .02 for i in range(totsame)]
    elif len(same_srs_width) < totsame: same_srs_width += [.7 + i * .02 for i in range(totsame - len(same_srs_width))]
    if isinstance(segs, pd.DataFrame): pass
    elif isinstance(segs, int):
        tchunk = (same_srs[0].index[-1] - same_srs[0].index[0]) / segs
        trange = [same_srs[0].index[0] + tchunk * i for i in range(segs)] + [same_srs[0].index[-1]]
        segs = pd.DataFrame([trange[:-1], trange[1:]], index = ['start', 'stop']).T
    else: segs = pd.DataFrame([[same_srs[0].index[0], same_srs[0].index[-1]]], columns = ['start', 'stop'])
    segs.reset_index(drop = True, inplace = True)
    totsp = len(spans)
    totline = len(lines)
   
    if isinstance(linestyle, list):
        if totline > 0:
            if len(linestyle) < totline: linestyle += ['-' for _ in range(totline - len(linestyle))]
    elif isinstance(linestyle, str): linestyle = [linestyle for _ in range(totline)]
    else: input_variable_error('Invalid line style! linestyle should be str or list of str with equal length of lines')
    if isinstance(linewidth, list):
        if totline > 0:
            if len(linewidth) < totline: linewidth = [1 for _ in range(totline - len(linewidth))]
    elif isinstance(linewidth, int) or isinstance(linewidth, float): linewidth = [linewidth for _ in range(totline)]
    else: input_variable_error('Invalid line width! linewidth should be int/float or list of int/float with equal length of lines')
   
    nrows = segs.shape[0]
    # colors = ['darkcyan', 'coral', 'darkslateblue', 'limegreen', 'crimson', 'purple', 'blue', 'khaki', 'chocolate', 'forestgreen']
    colors = get_named_colors()
    spcolors = ['darkcyan', 'coral', 'purple', 'red', 'khaki', 'gray', 'darkslateblue', 'limegreen', 'red', 'blue']
    lcolors = ['crimson', 'darkcyan' , 'darkolivegreen', 'palevioletred', 'indigo', 'chokolate', 'blue', 'forestgreen', 'grey', 'magenta']
    if totsame + totdif > len(colors): colors = get_color_from_cmap(totsame + totdif, 'rainbow')
    if totsp > len(spcolors): colors = get_color_from_cmap(totsp, 'rainbow')
    if totline > len(lcolors): colors = get_color_from_cmap(totline, 'rainbow')
    if spalpha == []: spalpha = [.3 for _ in range(totsp)]
    stamp_fl = True if isinstance(same_srs[0].index[0], pd.Timestamp) else False
    
    if fig_title == '': fig_title = 'Time Series Visualization'
    if axobj is None:
        fig, ax = plt.subplots(figsize = (fig_x, fig_y * nrows), nrows = nrows)
        fig.suptitle(fig_title, y = 1.03, fontsize = 20, fontweight = 'bold')
    else:
        ax = axobj
    if nrows == 1: ax = [ax]
    
    for i in range(nrows):
        lg = [[], []]
        st = segs['start'].iloc[i]
        ed = segs['stop'].iloc[i]
       
        # line plots
        for j in range(totline):
            ln = lines[j][(lines[j] >= st) & (lines[j] <= ed)]
            if ln.shape[0] > 0:
                for k in range(ln.shape[0]):
                    l = ax[i].axvline(ln.iloc[k], alpha = 1, color = lcolors[j], lw = linewidth[j], ls = linestyle[j])
                lg[0].append(l)
                lg[1].append(lines[j].name)
        
        # time series plots
        for j in range(totdif):
            tx = ax[i].twinx()
            l, = tx.plot(srs[j].loc[st : ed], color = colors[totsame + j], zorder = 30 + j * 4, marker = marker)
            lg[0].append(l)
            lg[1].append(srs[j].name)
            tx.spines['right'].set_position(('axes', 1 + j * spine_dist))
            tx.spines['right'].set_color(colors[totsame + j])
            tx.tick_params(axis = 'y', labelcolor = colors[totsame + j])
            tx.set_ylabel(name_separation(srs[j].name, name_thres), color = colors[totsame + j])
            tx.grid(False)
            if len(ylims) > 1:
                if len(ylims[1 + j]) == 2: tx.set_ylim(ylims[1 + j][0], ylims[1 + j][1])
        for j in range(totsame):
            l, = ax[i].plot(same_srs[j].loc[st : ed], color = colors[j], zorder = j * 4, lw = same_srs_width[j], marker = marker)
            lg[0].append(l)
            lg[1].append(same_srs[j].name)
        
        # span plots
        lgfl = [False for _ in range(totsp)]
        for j in range(totsp):
            for k in range(spans[j].shape[0]):
                spst = spans[j]['start'].iloc[k]
                sped = spans[j]['stop'].iloc[k]
                if spst < st and (sped > st and sped <= ed):
                    l = ax[i].axvspan(st, sped, color = spcolors[j], alpha = spalpha[j])
                    if not lgfl[j]:
                        lgfl[j] = True
                elif spst >= st and sped <= ed:
                    l = ax[i].axvspan(spst, sped, color = spcolors[j], alpha = spalpha[j])
                    if not lgfl[j]:
                        lgfl[j] = True
                elif (spst >= st and spst < ed) and sped > ed:
                    l = ax[i].axvspan(spst, ed, color = spcolors[j], alpha = spalpha[j])
                    if not lgfl[j]:
                        lgfl[j] = True
                elif spst <= st and sped >= ed:
                    l = ax[i].axvspan(st, ed, color = spcolors[j], alpha = spalpha[j])
                    if not lgfl[j]:
                        lgfl[j] = True
            if lgfl[j]:
                lg[0].append(l)
                lg[1].append(spans[j].columns.name)
       
        # finishing axis arrangements
        ax[i].set_xlabel(xlabel)
        ax[i].set_ylabel(ylabel if totsame > 1 else name_separation(same_srs[0].name, name_thres))
        if stamp_fl: ax[i].set_title('from %s to %s'%(st.strftime('%Y-%m-%d %H-%M-%S'), ed.strftime('%Y-%m-%d %H-%M-%S')), loc = 'right')
        ax[i].legend(lg[0], lg[1], loc = 3, bbox_to_anchor = (0, .98), ncol = len(lg[0]), fontsize = 10)
        ax[i].set_xlim(st, ed)
        ax[i].grid(True)
        plt.setp(ax[i].get_xticklabels(), rotation=xrot, ha=x_ha)
        if len(ylims) > 0:
            if len(ylims[0]) == 2: ax[i].set_ylim(ylims[0][0], ylims[0][1])
    if axobj is None:
        fig.tight_layout()
        if show: plt.show()
        if save and savepath is not None:
            os.makedirs(savepath, exist_ok=True)
            if fname == '': fname = fig_title.replace('.', '')
            fig.savefig('%s/%s.jpg'%(savepath, fname), bbox_inches = 'tight')
        plt.close()
    else:
        return ax[0]


def plot_heatmap(data, keep='both', rem_diag=False, cmap='gist_heat', cbar=True, stdz=False, annotate=False, fmt=None, vmin=None, vmax=None, center=None,
                 show=True, save=False, savepath=None, figsize=(30, 10), fig_title='', file_name='', xrot=90, axobj=None):
    """
    This function draws table like heatmap for a matrix data (should be pandas DataFrame)

    Inputs:
        :data: pandas DataFrame, data to be plotted as heatmap
        :stdz: bool, whether to standardize the data or not, default is False
        :keep: {'both', 'up', 'down'}, which side of the heatmap matrix to plot, necessary for correlation heatmap plot, default is 'both'
        :rem_diag: bool, whether to remove diagoanl if keep_only is not 'both', default is False
        :cmap: str, matplotlib colormap, default is 'gist_heat'
        :cbar: bool, show the colorbar with the heatmap or not
        :annotate: bool, whether to show the values or not
        :fmt: str, value format for printing if annotate is True, default is None
        :show: bool, show the heatmap or not, default is True
        :save: bool, save the figure or not, default is False
        :savepath: str, path for saving the figure, default is None
        :figsize: tuple, figure size, default is (30, 10)_
        :fig_title: str, title of the heatmap, default is 'Heatmap of {data.columns.name}'
        :file_name: str, name of the image as will be saved in savepath, default is fig_title
        :xrot: float, rotation angle of x-axis labels, default is 0
        :axobj: matplotlib axes object, to draw time series on this axes object plot. Default is None
    
    Outputs:
        :ax: if axobj is not None and axobj is provided as matplotlib axes object, then this function returns the axes object after drawing
    """

    ylb = data.index.name
    xlb = data.columns.name
    if stdz:
        data_std = data.std()
        data_std[data_std == 0] = 1
        for c in data.columns:
            if data[c].replace([0, 1], np.nan).dropna().shape[0] == 0:
                data_std.loc[c] = 1
        data = (data - data.mean()) / data_std
    if keep == 'up':
        k = 1 if rem_diag else 0
        data = data.where(np.triu(np.ones(data.shape), k = k).astype(bool))
    elif keep == 'down':
        k = -1 if rem_diag else 0
        data = data.where(np.tril(np.ones(data.shape), k = k).astype(bool))
    if axobj is None: fig, ax = plt.subplots(figsize = figsize)
    else: ax = axobj
    sns.heatmap(data, ax = ax, linewidths = 0, cbar = cbar, cmap = cmap, annot = annotate, fmt = fmt, center = center, vmin = vmin, vmax = vmax)
    ax.set_xlabel(xlb)
    ax.set_ylabel(ylb)
    ax.tick_params(axis = 'y', rotation = 0)
    ax.tick_params(axis = 'x', rotation = xrot)
    if fig_title == '': fig_title = 'Heatmap of %s'%data.columns.name if data.columns.name not in ['', None] else 'Heatmap'
    ax.set_title(fig_title)
    if axobj is None:
        fig.tight_layout()
        if show: plt.show()
        if save and savepath is not None:
            os.makedirs(savepath, exist_ok=True)
            if file_name == '': file_name = fig_title
            fig.savefig('%s/%s.jpg'%(savepath, file_name), bbox_inches = 'tight')
        plt.close()
    else:
        return ax


def data_gridplot(data, idf=[], idf_pref='', idf_suff='', diag='hist', bins=25, figsize=(16, 12), alpha=.7, 
                  s=None, lg_font='x-small', lg_loc=1, fig_title='', show_corr=True, show_stat=True, show=True, 
                  save=False, savepath=None, fname='', cmap=None):
    """
    This function creates a grid on nxn subplots showing scatter plot for each columns of 'data', n being the number of columns in 'data'.
    This function also shows histogram/kde/line plot for each columns along grid diagonal.
    The additional advantage is that we can separate each subplot data by a classifier item 'idf' which shows the class names for each row of data.
    This functionality is specially helpful to understand class distribution of the data.
    It also shows the correlation values (non-diagonal subplots) and general statistics (mean, median etc in diagonal subplots) for each subplot.

    Inputs:
        :data: pandas dataframe, list or numpy ndarray of rank-2, columns are considered as features
        :idf: pandas series with length equal to total number of samples in the data, works as clussifier of each sample data,\n
              specially useful in clustering, default is []
        :idf_pref: str, idf prefix, default is ''
        :idf_suff: str, idf suffix, default is ''
        :diag: {'hish', 'kde', 'plot'}, selection of diagonal plot, default is 'hist'
        :bins: int, number of bins for histogram plot along diagonal, default is 25
        :figsize: tuple, size of the whole figure, default is (16, 12)
        :alpha: float (0~1), transparency parameter for scatter plot, deafult is .7
        :s: float, point size for scatter plot, deafult is None
        :lg_font: float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}, legend font size, default is 'x-small'
        :lg_loc: int, location parameter for plot legend, deafult is 1
        :fig_title: str, titile of the whole figure, default is 'All Columns Grid Plot'
        :show_corr: bool, whether to show correlation value for each scatter plot, default is True
        :show_stat: bool, whether to show mean and std for each columns od data along diagonal plots, default is True
        :show: bool, whether to show the figure or not, default is True
        :save: bool, whether to save the graph or not, default is False
        :savepath: str, path to store the graph, default is None
        :fname: str, name used to store the graph in savepath, defualt is fig_title
        :cmap: str, matplotlib color map, default is None ('jet')

    Outputs:
        This function doesnt return anything.
    """
    
    if isinstance(idf, list) or isinstance(idf, np.ndarray): idf = pd.Series(idf, dtype = float)
    if isinstance(data, pd.Series): data = data.to_frame()
    elif isinstance(data, list) or isinstance(data, np.ndarray): data = pd.DataFrame(data, columns = ['col_%d'%i for i in range(len(data))])
    elif isinstance(data, pd.DataFrame): pass
    else: print('invalid data passed inside the function!')
    
    if data.shape[1] <= 1: raise ValueError('Data should include at least 2 columns!')
    if idf.shape[0] == 0: idf = pd.Series(np.zeros(data.shape[0]), index = data.index)
    colors = ['darkcyan', 'coral', 'limegreen', 'saddlebrown', 'grey', 'darkgoldenrod', 'forestgreen', 'purple',
              'crimson', 'cornflowerblue', 'darkslateblue', 'lightseagreen', 'darkkhaki', 'maroon', 'magenta', 'k']
    if data.shape[1] > len(colors) and cmap is None: cmap = 'jet'
    if cmap is not None: colors = get_color_from_cmap(data.shape[1], cmap = cmap)
   
    fig, ax = plt.subplots(figsize = figsize, nrows = data.shape[1], ncols = data.shape[1])
    if fig_title == '': fig_title = 'All Columns Grid Plot'
    fig.suptitle(fig_title, fontweight = 'bold', y = 1.03)
    col = data.columns
    idfs = sorted(idf.unique())
    idfslen = len(idfs)
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            coldata = data[col[[i, j]]].dropna()
            for k in range(idfslen):
                idx = idf[idf == idfs[k]].index.intersection(coldata.index)
                if idx.shape[0] > 0:
                    color = colors[j] if idfslen == 1 else colors[k]
                    if i == j:
                        kdy = plot_diag(coldata.iloc[:, 0].loc[idx], bins, ax[i, j], diag, color, '%s%s%s'%(idf_pref, idfs[k], idf_suff))
                        if k == 0: txt = 'mean: %.3f, std: %.3f'%(np.nanmean(data[col[i]]), np.nanstd(data[col[i]])) if show_stat else ''
                        if diag == 'plot':
                            txt_x = coldata.index.min()
                            txt_y = coldata.iloc[:, 0].min()
                        elif diag == 'hist':
                            txt_x = coldata.iloc[:, 0].min()
                            txt_y = .01
                        elif diag == 'kde':
                            txt_x = coldata.iloc[:, 0].min()
                            txt_y = kdy
                       
                    else:
                        if k == 0: txt = 'corr: %.3f'%coldata.corr().values[0, 1] if show_corr else ''
                        ax[i, j].scatter(coldata[col[j]].loc[idx].values, coldata[col[i]].loc[idx].values, color = color, alpha = alpha, s = s, label = '%s%s%s'%(idf_pref, idfs[k], idf_suff))
                        txt_x = coldata[col[j]].min()
                        txt_y = coldata[col[i]].min()
            if idfslen > 1: ax[i, j].legend(fontsize = lg_font, loc = lg_loc)
            ax[i, j].text(txt_x, txt_y, s = txt, ha = 'left', va = 'bottom')
            if i == 0: ax[i, j].set_title(col[j])
            if j == 0: ax[i, j].set_ylabel(col[i])
            ax[i, j].grid(True)
    fig.tight_layout()
    if show: plt.show()
    if save and savepath != '':
        os.makedirs(savepath, exist_ok=True)
        if fname == '': fname = fig_title
        fig.savefig('%s/%s.jpg'%(savepath, fname), bbox_inches = 'tight')
    plt.close()

# associative function for data_gridplot()
def plot_diag(arr, bins, ax, diag, color, label):
    if diag == 'hist': ax.hist(arr.values, bins = bins, color = color, label = label)
    elif diag == 'kde':
        x = np.linspace(arr.min(), arr.max(), 500)
        ax.plot(x, gaussian_kde(arr.values)(x), color = color, label = label)
        return np.min(gaussian_kde(arr.values)(x))
    elif diag == 'plot': ax.plot(arr.index, arr.values, color = color, label = label)
    else: print('Invalid diagonal plot type!')
    return None


def get_color_from_cmap(n, cmap='jet', rng=[.05, .95]):
    """
    this function gets color from matplotlib colormap

    Inputs:
        :n: int, number of colors you want to create
        :cmap: str, matplotlib colormap name, default is 'jet'
        :rng: array, list or tuple of length 2, edges of colormap, default is [.05, .95]
    
    Outputs:
        It returns a list of colors from the selected colormap
    """
    
    return [get_cmap(cmap)(i) for i in np.linspace(rng[0], rng[1], n)]


def get_named_colors():
    """
    This function returns 36 custom selected CSS4 named colors available in matplotlib

    Outputs:
        list of colors/color_names available in matplotlib
    """
    colors = ['darkcyan', 'crimson', 'coral', 'forestgreen', 'purple', 'magenta', 'khaki', 'maroon',
              'darkslategray', 'brown', 'gray', 'darkorange', 'mediumseagreen', 'royalblue', 'midnightblue',
              'turquoise', 'rosybrown', 'darkgoldenrod', 'olive', 'saddlebrown', 'darkseagreen', 'deeppink',
              'aqua', 'lawngreen', 'tab:red', 'gold', 'tan', 'peru', 'violet', 'thistle', 'steelblue',
              'darkkhaki', 'chocolate', 'mediumspringgreen', 'rebeccapurple', 'tomato']
    return colors


def plot_table(_data, cell_width=2.5, cell_height=0.625, font_size=14, header_color='#003333', row_colors=['whitesmoke', 'w'], 
    edge_color='w', bbox=[0, 0, 1, 1], header_cols=0, index_cols=0, fig_name='Table', save=False, show=True, savepath=None, axobj=None, 
    **kwargs):
    """
    This function creates figure for a pandas dataframe table. It drops the dataframe index at first. 
    So, if index is necessary to show, keep it as a dataframe column.
    
    Inputs:
        :_data: pandas dataframe, which will be used to make table and save as a matplotlib figure. Index is removed in the begining.\n
               So, if index is necessary to show, keep it as a dataframe column.
        :cell_width: float, each cell width, default is 2.5
        :cell_height: float, each cell height, default is 0.625
        :font_size: float, font size for table values, default is 14
        :header_color: str or rgb color code, column and index color, default is '#003333'
        :row_colors: str or rgb color code, alternating colors to separate consecutive rows, default is ['whitesmoke', 'w']
        :edge_color: str or rgb color code, cell border color, default is 'w'
        :bbox: list of four floats, edges of the table to cover figure, default is [0,0,1,1]
        :header_cols: int, number of initial rows to cover column names, default is 0
        :index_cols: int, number of columns to cover index of dataframe, default is 0 
        :fig_name: str, figure name to save the figure when save is True, default is 'Table'
        :save: bool, whether to save the figure, default is False
        :show: bool, whether to show the plot, default is True
        :savepath: path to store the figure, default is current directory from where the code is being run ('.')
        :ax: matplotlib axis to include table in another figure, default is None
    
    Outputs:
        :ax: matplotlib axes object used for plotting, if provided as axobj
    """
    
    data = _data.reset_index(drop=False)
    if axobj is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([cell_width, cell_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    else:
        ax = axobj
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for rc, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if rc[0] == 0 or rc[1] < header_cols or rc[1] == 0 or rc[0] < index_cols:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[rc[0]%len(row_colors) ])
    if axobj is None:
        fig.tight_layout()
        if save and savepath is not None:
            os.makedirs(savepath, exist_ok=True)
            fig.savefig('%s/%s.png'%(savepath, fig_name), bbox_inches='tight')
        if show: plt.show()
        else: plt.close()
    else:
        return ax


def plot_class_score(score, confmat, xrot=0, figure_dir=None, figtitle=None, figsize=(15, 5), show=False, cmap='Blues'):
    """
    This function generates figure showing the classification score and confusion matrix as tables side by side.

    Inputs:
        :score: pandas DataFrame, contains score matrix values of all classes for all matrices
        :confmat: pandas DataFrame, contains confusion matrix values for all classes
        :xrot: float, rotation angle for x-axis labels (expected to be class names). Default is 0.
        :figure_dir: str or None, path to the directory where the figure will be saved. Default is None (means it wont be saved)
        :figsize: tuple of floats (xsize, ysize) sets the matplotlib figure size. Default is (15, 5)
        :show: bool, whether to show the figure or not, default is False
        :cmap: str, colormap name of the score and confusion matrix tables
    """

    fig, ax = plt.subplots(figsize=figsize, ncols=2)
    if figtitle is None:
        figtitle = 'Classification score'
    fig.suptitle(figtitle, y=1.04, fontweight='bold', fontsize=12)
    ax[0] = plot_heatmap(score, axobj=ax[0], annotate=True, fmt='.3f', vmin=0, vmax=1, fig_title='Score Matrix', cmap=cmap, xrot=xrot)
    ax[1] = plot_heatmap(confmat, axobj=ax[1], annotate=True, fmt='d', fig_title='Confusion Matrix', cmap=cmap, xrot=xrot)
    fig.tight_layout()
    if figure_dir is not None:
        fig.savefig('%s/%s.png'%(figure_dir, figtitle), bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def plot_regression_score(true, pred, figure_dir=None, figtitle=None, figsize=(10, 5), show=False,
                          point_color='darkcyan', s=8, metrics={}):
    """
    This function generates true-value VS predicted value scatter plot for regression problems.

    Inputs:
        :true: 1D list, numpy array etc. containing true values
        :pred: 1D list, numpy array etc. containing predicted values
        :figure_dir: str or None, path to the directory where the figure will be saved. Default is None (means it wont be saved)
        :figsize: tuple of floats (xsize, ysize) sets the matplotlib figure size. Default is (10, 5)
        :show: bool, whether to show the figure or not, default is False
        :point_color: str, color of the scatter points, default is 'darkcyan'
        :s: float, scatter point size, default is 8
    """

    if not isinstance(true, np.ndarray):
        true = np.array(true)
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(true, pred, color=point_color, s=s)
    _min = np.min([true.min(), pred.min()])
    _max = np.max([true.max(), pred.max()])
    ax.plot([_min, _max], [_min, _max], color='k', lw=2)
    
    ax.set_xlabel('true-label')
    ax.set_ylabel('prediction')
    if figtitle is None:
        figtitle = 'True-Label VS Prediction Scatter plot'
    savetitle = figtitle
    if len(metrics) > 0:
        metstr = ''
        for i, name in enumerate(metrics):
            if i > 0:
                metstr += ',   '
            metstr += str(name) + ' : ' + str(metrics[name])
        figtitle += '\n%s'%metstr
    ax.set_title(figtitle, fontweight='bold')
    fig.tight_layout()
    if figure_dir is not None:
        fig.savefig('%s/%s.png'%(figure_dir, savetitle), bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
