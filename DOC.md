# Documentation of all functions and classes
#### <center>(ordered alphabetically)</center>





## class_result
<body>
############ CLASS_RESULT  ##############

classification result with confusion matrix

ARGUMENTS

y: True labels

pred: Predictions

out_confus: returns confusion matrix if set True

RETURNS

result: DataFrame containing precision, recall and f1 scores for all labels

con_mat: DataFrame containing confusion matrix, depends on out_confus
</body>

## data_gridplot
<body>
########### DATA_GRIDPLOT  #############
this function generates scatter plots between every 2 columns in the data set and distribuition or kde or time series plot along diagonal
data : pandas dataframe, list or numpy ndarray of rank-2, columns are considered as features
idf : pandas series with length equal to total number of samples in the data, works as clussifier of each sample data, specially useful in clustering, default is []
idf_pref : str, idf prefix, default is ''
idf_suff: str, idf suffix, default is ''
diag : {'hish', 'kde', 'plot'}, selection of diagonal plot, default is 'hist'
bins : int, number of bins for histogram plot along diagonal, default is 25
figsize : tuple, size of the whole figure, default is (16, 12)
alpha : float (0~1), transparency parameter for scatter plot, deafult is .7
s : float, point size for scatter plot, deafult is None
lg_font : float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}, legend font size, default is 'x-small'
lg_loc : int, location parameter for plot legend, deafult is 1
fig_title: str, titile of the whole figure, default is 'All Columns Grid Plot'
show_corr : bool, whether to show correlation value for each scatter plot, default is True
show_stat : bool, whether to show mean and std for each columns od data along diagonal plots, default is True
show : bool, whether to show the figure or not, default is True
save : bool, whether to save the graph or not, default is False
savepath : str, path to store the graph
fname : str, name used to store the graph in savepath
cnam : str, matplotlib color map
</body>


## feature_evaluator
<body>
########  FEATURE_EVALUATOR  ######################
data: should be a pandas dataframe containing label data inside it (it must contain only features and labels, no unnecessary columns are allowed)
label_name: list or str, containing name of the columns used as labels
label_type_num: list of bool, containing flag info if the labels are numerical or not (must be corresponding to label_name)
n_bin: int, expressing number of divisions of the label values. If any label is categorical, n_bin for that label will be equal to the number of categories, default is 40 for numerical labels
(here its okay to use labels before converting into one hot encoding, its even better not to convert into one hot)
is_all_num: bool, is a simplification flag indicates whether all the features are numerical or not
is_all_cat: bool, is a simplification flag indicates whether all the features are categorical or not
num_cols: list, containing name of numerical columns if needed
cat_cols: list, containing categorical feature names if needed
cmap: str, is the matplotlib colormap name, default is 'gist_rainbow'
fig_width: float, is the width of figure, default is 30
fig_height: float, is the height of figure, default is 5
rotation: float, rotational angle of x axis labels of the figure, default is 40
This function tries to calculate the importance of features by some statistical approaches
</body>




## Filters
<body>
########## init arguments ################
T : int/float, indicating the sampling period of the signal, must be in seconds (doesnt have any default values)
n : int, indicating number of fft frequency bins, default is 1000
</body>

#### vis_spectrum
<body>
######### vis_spectrum arguments ##############
sr : pandas series, indicating the time series signal you want to check the spectrum for
f_lim : python list of len 2, indicating the limits of visualizing frequency spectrum
see_neg : bool, flag indicating whether to check negative side of the spectrum or not
show : bool, whether to see the figure or not
save : bool, whether to save the figure or not
savepath : str, path for saving the figure
figsize : tuple, size of the figure plotted to show the fft version of the signal
</body>

#### apply
<body>
################## apply arguments ######################
sr : pandas series, indicating the time series signal you want to apply the filter on
filt_type : str, indicating the type of filter you want to apply on sr. {'lp', 'low_pass', 'low pass', 'lowpass'} for applying    
low pass filter and similar for 'high pass', 'band pass' and 'band stop' filters
f_cut : int/float/pylist/numpy array-n indicating cut off frequencies. Please follow the explanations bellow
for lowpass/highpass filters, it must be int/float
for bandpass/bandstop filters, it must be a list or numpy array of length divisible by 2
order : int, filter order, the more the values, the sharp edges at the expense of complexer computation
response : bool, whether to check the frequency response of the filter or not
plot : bool, whether to see the spectrum and time series plot of the filtered signal or not
f_lim : pylist of length 2, frequency limit for the plot
</body>


## get_color_from_cmap
<body>
######## GET_COLOR_FROM_CMAP  ###########
this function gets color from colormap
n : int, number of colors you want to create
cmap : str, matplotlib colormap name, default is 'jet'
rng : array, list or tuple of length 2, edges of colormap, default is [.05, .95]
</body>


## get_edges_from_ts
<body>
############ GET_EDGES ##############
this function finds the edges specifically indices of start and end of certain regions by thresholding the desired time series data
sr : pandas Series, time series data with proper timestamp as indices of the series
th_method : {'mode', 'median', 'mean'}, method of calculating threshold
th_factor : flaot/int, multiplication factor to be multiplied with th_method value to calculate threshold
th : int/flaot, threshold value inserted as argument
del_side : {'up', 'down'}, indicating which side to be removed to get edges
Note: the algorithm starts recording when it exceeds the threshold value, so open interval system.
</body>

## get_spectrogram
<body>
######### find the spectrogram of a time series signal ############
ts_sr : pandas.Series object containing the time series data, the series should contain its name as ts_sr.name
fs : int/flaot, sampling frequency of the signal, default is 1
win : (str, float), tuple of window parameters, default is (tukey, .25)
nperseg : int, number of data in each segment of the chunk taken for STFT, default is 256
noverlap : int, number of data in overlapping region, default is (nperseg // 8)
mode : str, type of the spectrogram output, default is power spectral density('psd')
figsize : tuple, figsize of the plot, default is (30, 6)
vis_frac : float or a list of length 2, fraction of the frequency from lower to higher, you want to visualize, default is 1(full range)
ret_sxx : bool, whehter to return spectrogram dataframe or not, default is False
show: bool, whether to show the plot or not, default is True
save : bool, whether to save the figure or not, default is False
savepath : str, path to save the figure, default is ''
fname : str, name of the figure to save in the savepath, default is fig_title
</body>



## grouped_mode
<body>
###########  GROUPED_MODE  ############
data : pandas Series, list, numpy ndarray - must be 1-D
bins : int, list or ndarray, indicates bins to be tried to calculate mode value
neglect_values : list, ndarray, the values inside the list will be removed from data
neglect_above : float, values above this will be removed from the data
neglect_beloow : float, values bellow this will be removed from the data
neglect_quan : 0 < float < 1 , percentile range which will be removed from both sides from data distribution
</body>



## moving_slope
<body>
moving slope is a function which calculates the slope inside a window for a variable
df : pandas DataFrame or Series, contains time series columns
fs : str, the base sampling period of the time series, default is the mode value of the difference in time between two consecutive samples
win : int, window length, deafult is 60
take_abs : bool, indicates whether to take the absolute value of the slope or not
returns the pandas DataFrame containing resultant windowed slope values
</body>


## name_separation
<body>
string : str, the input string
maxlen : int, maximum allowable length for each line
</body>


## normalize
<body>
###########  NORMALIZE  #############
data : pandas series, dataframe, list or numpy ndarray, input data to be standardized
zero_range : float, value used to replace range values in case range is 0 for any column
</body>


## one_hot_encoding
<body>
############  ONE_HOT_ENCODING  ###################
arr : numpy array or list or pandas series, containing all class labels with same size as data set
class_label(optional) : numpy array or list, list or pandas series, containing only class labels
out_type : {'ndarray', 'series', 'list'}, data type of the output
</body>


## plot_heatmap
<body>
data : pandas DataFrame, data to be plotted as heatmap
stdz : bool, whether to standardize the data or not, default is False
keep_only : {'both', 'up', 'down'}, which side of the heatmap matrix to plot, necessary for correlation heatmap plot, default is 'both'
rem_diag : bool, whether to remove diagoanl if keep_only is not 'both', default is False
cmap : str, matplotlib colormap, default is 'gist_heat'
cbar : bool, show the colorbar with the heatmap or not
show : bool, show the heatmap or not
save : bool, save the figure or not
savepath : str, path for saving the figure
figsize : tuple, figure size
fig_title : str, title of the heatmap, default is 'Heatmap of {data.columns.name}'
file_name : str, name of the image as will be saved in savepath, default is fig_title
lbfactor : float/int, factor used for scaling the plot labels
</body>


## plot_time_series
<body>
################ PLOT_TIME_SERIES  ####################
This function plots time series data along with much additional information if provided as argument

same_srs : list of pandas series holding the variables which share same axis in matplotlib plot
srs : list of pandas series holding the variables which share different axes, default is []
segs : pandas dataframe with two columns 'start' and 'stop' indicating the start and stop of each axis plot segment
same_srs_width : list of flaots indicating each line width of corresponding same_srs time series data, must be same size as length of same_srs, default is []
spans : list of pandas dataframe indicating the 'start' and 'stop' of the span plotting elements
lines : list of pandas series where the values will be datetime of each line position, keys will be just serials
linestyle : list of str, marker for each line, '' indicates continuous straight line
linewidth : list of constant, line width for each line
fig_title : title of the entire figure
show : bool, indicating whether to show the figure or not
save bool, indicating whether to save the figure or not
savepath : str, location of the directory where the figure will be saved
fname  : str, figure name when the figure is saved
spine_dist : constant indicating distance of the right side spines from one another if any
spalpha : list of constants indicating alpha values for each of the span in spans
ylims : list of lists, indicating y axis limits in case we want to keep the limit same for all subplots
name_thres : maximum allowed characters in one line for plot labels
</body>


## ProgressBar

<body>
this is a custom designed progress bar for checking the loop timing. The user should follow this approach to make it work 
with ProgressBar(arr, desc = 'intro of arr', perc = 5) as pbar:
   for i in arr:
       'your code/task inside the loop'
       pbar.inc()
######## arguments  ###########
arr : iterable, it is the array you will use to run the loop, it can be range(10) or any numpy array or python list or any other iterator
desc(optional) : str, description of the loop, default - 'progress'
barlen : int, length of the progress bar, default is 40
front space : int, allowed space for description, default is 20
tblink_max : float/int indicates maximum interval in seconds between two adjacent blinks, default is .4
tblink_min : float/int indicates minimum interval in seconds between two adjacent blinks, default is .18
</body>



## rsquare_rmse
<body>
determination of coefficient R square value and root mean squared error value for regression evaluation
y: true values
pred: prediction values
</body>



## SplitDataset
<body>
initialize the class
data: pandas dataframe or series or numpy array of dimension #samples X other dimensions with rank 2 or higher
label: pandas dataframe or series or numpy array with dimension #samples X #classes with rank 2
index : numpy array, we can explicitly insert indices of the data set and labels
test_ratio : float, ratio of test data
np_seed : int, numpy seed used for random shuffle
same_ratio : bool, should be only true if the labels are classification labels and we want to keep the test and validation ratio same for all classes
make_onehot : bool, if True, the labels will be converted into one hot encoded format
</body>

### random_split
<body>
############ RANDOM_SPLIT  ################
this function is for simple random split
val_ratio : float, validation data set ratio, default is .15
returns data, label and indices for train, validation and test data sets as pydict
</body>

### cross_validation_split
<body>
###############  CROSS_VALIDATION_SPLIT  ####################
this function is prepared to run cross validation using proper data and labels
fold : int, number of folds being applied to the data, default is 5
returns data, label and indices for train, validation and test data sets as pydict
</body>

### sequence_split
<body>
######### SEQUENCE_SPLIT  ################
seq_len : int, length of each sequence
val_ratio : float, validation data set ratio, deafult is .15
data_stride : int, indicates the number of shift between two adjacent example data to prepare the data set, default is 1
label_shift : temporal difference between the label and the corresponding data's last time step, default is 1
split_method : {'train_val', 'multiple_train_val'}, default is 'multiple_train_val'
sec : int or pandas dataframe with columns naming 'start' and 'stop', number of sections for multiple_train_val method splitting (basically train_val splitting is multiple_train_val method with sec = 1), deafult is 1
dist : int, number of data to be leftover between two adjacent sec, default is 0
returns data, label and indices for train, validation and test data sets as pydict
</body>


## standardize
<body>
############  STANDARDIZE  #############
data : pandas series, dataframe, list or numpy ndarray, input data to be standardized
zero_std : float, value used to replace std values in case std is 0 for any column
</body>
