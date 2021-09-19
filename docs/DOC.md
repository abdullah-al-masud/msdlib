# Documentation of all functions and classes

### NB: documentation for dataset.py is [here](https://github.com/abdullah-al-masud/msdlib/blob/master/dataset_DOC.md)


#### <center>(ordered alphabetically)</center>



## class_result
<body>
############ CLASS_RESULT  ##############<br>

classification result with confusion matrix<br>
ARGUMENTS<br>
y: True labels, a vector holding class indices like 0, 1, 2 .. etc.<br>
pred: Predictions, a vector containing prediction opt for class indices similar to True labels<br>
out_confus: returns confusion matrix if set True<br>
RETURNS<br>
result: DataFrame containing precision, recall and f1 scores for all labels<br>
con_mat: DataFrame containing confusion matrix, depends on out_confus<br>
</body>

## data_gridplot
<body>
########### DATA_GRIDPLOT  #############<br>

this function generates scatter plots between every 2 columns in the data set and distribuition or kde or time series plot along diagonal<br>
data : pandas dataframe, list or numpy ndarray of rank-2, columns are considered as features<br>
idf : pandas series with length equal to total number of samples in the data, works as clussifier of each sample data, specially useful in clustering, default is []<br>
idf_pref : str, idf prefix, default is ''<br>
idf_suff: str, idf suffix, default is ''<br>
diag : {'hish', 'kde', 'plot'}, selection of diagonal plot, default is 'hist'<br>
bins : int, number of bins for histogram plot along diagonal, default is 25<br>
figsize : tuple, size of the whole figure, default is (16, 12)<br>
alpha : float (0~1), transparency parameter for scatter plot, deafult is .7<br>
s : float, point size for scatter plot, deafult is None<br>
lg_font : float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}, legend font size, default is 'x-small'<br>
lg_loc : int, location parameter for plot legend, deafult is 1<br>
fig_title: str, titile of the whole figure, default is 'All Columns Grid Plot'<br>
show_corr : bool, whether to show correlation value for each scatter plot, default is True<br>
show_stat : bool, whether to show mean and std for each columns od data along diagonal plots, default is True<br>
show : bool, whether to show the figure or not, default is True<br>
save : bool, whether to save the graph or not, default is False<br>
savepath : str, path to store the graph<br>
fname : str, name used to store the graph in savepath<br>
cnam : str, matplotlib color map<br>
</body>


## each_row_max
############## EACH_ROW_MAX ###############<br>
this function gets the maximum values and corresponding columns for each row of a matrix<br>
data : list of lists/numpy ndarray or pandas dataframe, matrix data from where max values will be calculated<br>
returns same data with two new columns with max values and corresponding column names<br>


## feature_evaluator
<body>
########  FEATURE_EVALUATOR  ######################<br>
This function tries to calculate the importance of features by some statistical approaches<br>
data: should be a pandas dataframe containing label data inside it (it must contain only features and labels, no unnecessary columns are allowed)<br>
label_name: list, containing name of the columns used as labels<br>
label_type_num: list of bool, containing flag info if the labels are numerical or not (must be corresponding to label_name)<br>
n_bin: int, expressing number of divisions of the label values. If any label is categorical, n_bin for that label will be equal to the number of categories, default is 40 for numerical labels<br>
(here its okay to use labels before converting into one hot encoding, its even better not to convert into one hot)<br>
is_all_num: bool, is a simplification flag indicates whether all the features are numerical or not<br>
is_all_cat: bool, is a simplification flag indicates whether all the features are categorical or not<br>
num_cols: list, containing name of numerical columns if needed<br>
cat_cols: list, containing categorical feature names if needed<br>
cmap: str, is the matplotlib colormap name, default is 'gist_rainbow'<br>
fig_width: float, is the width of figure, default is 30<br>
fig_height: float, is the height of figure, default is 5<br>
rotation: float, rotational angle of x axis labels of the figure, default is 40<br>
show : bool, whether to show the graph or not, default is True<br>
save : bool, whether to save the figure or not, default is False<br>
savepath : str, path where the data will be saved, default is None<br>
fname : str, filename which will be used for saving the figure, default is None<br>
fig_title : str, title of the figure, default is 'Feature Confidence Evaluation for categorical/Numerical Features'<br>
</body>




## Filters
<body>
########## init arguments ################<br>
T : int/float, indicating the sampling period of the signal, must be in seconds (doesnt have any default values)<br>
n : int, indicating number of fft frequency bins, default is 1000<br>
</body>

#### vis_spectrum
<body>
######### vis_spectrum arguments ##############<br>
sr : pandas series, indicating the time series signal you want to check the spectrum for<br>
f_lim : python list of len 2, indicating the limits of visualizing frequency spectrum<br>
see_neg : bool, flag indicating whether to check negative side of the spectrum or not<br>
show : bool, whether to see the figure or not<br>
save : bool, whether to save the figure or not<br>
savepath : str, path for saving the figure<br>
figsize : tuple, size of the figure plotted to show the fft version of the signal<br>
</body>

#### apply
<body>
################## apply arguments ######################<br>
sr : pandas series, indicating the time series signal you want to apply the filter on<br>
filt_type : str, indicating the type of filter you want to apply on sr. {'lp', 'low_pass', 'low pass', 'lowpass'} for applying    
low pass filter and similar for 'high pass', 'band pass' and 'band stop' filters<br>
f_cut : int/float/pylist/numpy array-n indicating cut off frequencies. Please follow the explanations bellow<br>
for lowpass/highpass filters, it must be int/float<br>
for bandpass/bandstop filters, it must be a list or numpy array of length divisible by 2<br>
order : int, filter order, the more the values, the sharp edges at the expense of complexer computation<br>
response : bool, whether to check the frequency response of the filter or not<br>
plot : bool, whether to see the spectrum and time series plot of the filtered signal or not<br>
f_lim : pylist of length 2, frequency limit for the plot<br>
</body>


## get_color_from_cmap
<body>
######## GET_COLOR_FROM_CMAP  ###########<br>
this function gets color from colormap<br>
n : int, number of colors you want to create<br>
cmap : str, matplotlib colormap name, default is 'jet'<br>
rng : array, list or tuple of length 2, edges of colormap, default is [.05, .95]<br>
</body>


## get_edges_from_ts
<body>
############ GET_EDGES ##############<br>
this function finds the edges specifically indices of start and end of certain regions by thresholding the desired time series data<br>
sr : pandas Series, time series data with proper timestamp as indices of the series<br>
th_method : {'mode', 'median', 'mean'}, method of calculating threshold<br>
th_factor : flaot/int, multiplication factor to be multiplied with th_method value to calculate threshold<br>
th : int/flaot, threshold value inserted as argument<br>
del_side : {'up', 'down'}, indicating which side to be removed to get edges<br>
Note: the algorithm starts recording when it exceeds the threshold value, so open interval system.<br>
</body>


## get_named_colors
<body>
###############  GET_NAMED_COLORS  #################33333 <br>
this function returns namy css4 named colors available in matplotlib

## get_spectrogram
<body>
######### find the spectrogram of a time series signal ############<br>
ts_sr : pandas.Series object containing the time series data, the series should contain its name as ts_sr.name<br>
fs : int/flaot, sampling frequency of the signal, default is 1<br>
win : (str, float), tuple of window parameters, default is (tukey, .25)<br>
nperseg : int, number of data in each segment of the chunk taken for STFT, default is 256<br>
noverlap : int, number of data in overlapping region, default is (nperseg // 8)<br>
mode : str, type of the spectrogram output, default is power spectral density('psd')<br>
figsize : tuple, figsize of the plot, default is (30, 6)<br>
vis_frac : float or a list of length 2, fraction of the frequency from lower to higher, you want to visualize, default is 1(full range)<br>
ret_sxx : bool, whehter to return spectrogram dataframe or not, default is False<br>
show: bool, whether to show the plot or not, default is True<br>
save : bool, whether to save the figure or not, default is False<br>
savepath : str, path to save the figure, default is ''<br>
fname : str, name of the figure to save in the savepath, default is fig_title<br>
</body>



## grouped_mode
<body>
###########  GROUPED_MODE  ############<br>
data : pandas Series, list, numpy ndarray - must be 1-D<br>
bins : int, list or ndarray, indicates bins to be tried to calculate mode value<br>
neglect_values : list, ndarray, the values inside the list will be removed from data<br>
neglect_above : float, values above this will be removed from the data<br>
neglect_beloow : float, values bellow this will be removed from the data<br>
neglect_quan : 0 < float < 1 , percentile range which will be removed from both sides from data distribution<br>
</body>



## moving_slope
<body>
moving slope is a function which calculates the slope inside a window for a variable<br>
df : pandas DataFrame or Series, contains time series columns<br>
fs : str, the base sampling period of the time series, default is the mode value of the difference in time between two consecutive samples<br>
win : int, window length, deafult is 60<br>
take_abs : bool, indicates whether to take the absolute value of the slope or not<br>
returns the pandas DataFrame containing resultant windowed slope values<br>
</body>


## name_separation
<body>
string : str, the input string<br>
maxlen : int, maximum allowable length for each line<br>
</body>


## normalize
<body>
###########  NORMALIZE  #############<br>
data : pandas series, dataframe, list or numpy ndarray, input data to be standardized<br>
zero_range : float, value used to replace range values in case range is 0 for any column<br>
method : {'zero_mean', 'min_max_0_1', 'min_max_-1_1'}<br>
</body>


## one_hot_encoding
<body>
############  ONE_HOT_ENCODING  ###################<br>
arr : numpy array or list or pandas series, containing all class labels with same size as data set<br>
class_label(optional) : numpy array or list, list or pandas series, containing only class labels<br>
out_type : {'ndarray', 'series', 'list'}, data type of the output<br>
</body>


## plot_heatmap
<body>
data : pandas DataFrame, data to be plotted as heatmap<br>
stdz : bool, whether to standardize the data or not, default is False<br>
keep : {'both', 'up', 'down'}, which side of the heatmap matrix to plot, necessary for correlation heatmap plot, default is 'both'<br>
rem_diag : bool, whether to remove diagoanl if keep is not 'both', default is False<br>
cmap : str, matplotlib colormap, default is 'gist_heat'<br>
cbar : bool, show the colorbar with the heatmap or not<br>
annotate : bool, whether to show the values or not<br>
fmt : str, value format for printing if annotate is True<br>
show : bool, show the heatmap or not<br>
save : bool, save the figure or not<br>
savepath : str, path for saving the figure<br>
figsize : tuple, figure size<br>
fig_title : str, title of the heatmap, default is 'Heatmap of {data.columns.name}'<br>
file_name : str, name of the image as will be saved in savepath, default is fig_title<br>
lbfactor : float/int, factor used for scaling the plot labels<br>
</body>


## plot_time_series
<body>
################ PLOT_TIME_SERIES  ####################<br>
This function plots time series data along with much additional information if provided as argument<br>

same_srs : list of pandas series holding the variables which share same axis in matplotlib plot<br>
srs : list of pandas series holding the variables which share different axes, default is []<br>
segs : pandas dataframe with two columns 'start' and 'stop' indicating the start and stop of each axis plot segment<br>
same_srs_width : list of flaots indicating each line width of corresponding same_srs time series data, must be same size as length of same_srs, default is []<br>
spans : list of pandas dataframe indicating the 'start' and 'stop' of the span plotting elements<br>
lines : list of pandas series where the values will be datetime of each line position, keys will be just serials<br>
linestyle : list of str, marker for each line, '' indicates continuous straight line<br>
linewidth : list of constant, line width for each line<br>
fig_title : title of the entire figure<br>
show : bool, indicating whether to show the figure or not<br>
save bool, indicating whether to save the figure or not<br>
savepath : str, location of the directory where the figure will be saved<br>
fname  : str, figure name when the figure is saved<br>
spine_dist : constant indicating distance of the right side spines from one another if any<br>
spalpha : list of constants indicating alpha values for each of the span in spans<br>
ylims : list of lists, indicating y axis limits in case we want to keep the limit same for all subplots<br>
name_thres : maximum allowed characters in one line for plot labels<br>
fig_x: float, horizontal length of the figure, default is 30<br>
fig_y: float, vertical length of each row of plot, default is 3<br>
marker: str, marker for time series plots, default is None<br>
xlabel : str, label name for x axis for each row, default is 'Time'<br>
ylabel : str, label name for y axis for each row, default is 'Data value'<br>
</body>


## ProgressBar

<body>
this is a custom designed progress bar for checking the loop timing. The user should follow this approach to make it work <br>
with ProgressBar(arr, desc = 'intro of arr', perc = 5) as pbar:<br>
   for i in arr:<br>
       'your code/task inside the loop'<br>
       pbar.inc()<br>
######## arguments  ###########<br>
arr : iterable, it is the array you will use to run the loop, it can be range(10) or any numpy array or python list or any other iterator<br>
desc(optional) : str, description of the loop, default - 'progress'<br>
barlen : int, length of the progress bar, default is 40<br>
front space : int, allowed space for description, default is 20<br>
tblink_max : float/int indicates maximum interval in seconds between two adjacent blinks, default is .4<br>
tblink_min : float/int indicates minimum interval in seconds between two adjacent blinks, default is .18<br>
</body>



## rsquare_rmse
<body>
determination of coefficient R square value and root mean squared error value for regression evaluation<br>
y: true values<br>
pred: prediction values<br>
</body>



## SplitDataset
<body>
initialize the class<br>
data: pandas dataframe or series or numpy array of dimension #samples X other dimensions with rank 2 or higher<br>
label: pandas dataframe or series or numpy array with dimension #samples X #classes with rank 2<br>
index : numpy array, we can explicitly insert indices of the data set and labels<br>
test_ratio : float, ratio of test data<br>
np_seed : int, numpy seed used for random shuffle<br>
same_ratio : bool, should be only true if the labels are classification labels and we want to keep the test and validation ratio same for all classes<br>
make_onehot : bool, if True, the labels will be converted into one hot encoded format<br>
</body>

#### random_split
<body>
############ RANDOM_SPLIT  ################<br>
this function is for simple random split<br>
val_ratio : float, validation data set ratio, default is .15<br>
returns data, label and indices for train, validation and test data sets as pydict<br>
</body>

#### cross_validation_split
<body>
###############  CROSS_VALIDATION_SPLIT  ####################<br>
this function is prepared to run cross validation using proper data and labels<br>
fold : int, number of folds being applied to the data, default is 5<br>
returns data, label and indices for train, validation and test data sets as pydict<br>
</body>

#### sequence_split
<body>
######### SEQUENCE_SPLIT  ################<br>
seq_len : int, length of each sequence<br>
val_ratio : float, validation data set ratio, deafult is .15<br>
data_stride : int, indicates the number of shift between two adjacent example data to prepare the data set, default is 1<br>
label_shift : temporal difference between the label and the corresponding data's last time step, default is 1<br>
split_method : {'train_val', 'multiple_train_val'}, default is 'multiple_train_val'<br>
sec : int or pandas dataframe with columns naming 'start' and 'stop', number of sections for multiple_train_val method splitting (basically train_val splitting is multiple_train_val method with sec = 1), deafult is 1<br>
dist : int, number of data to be leftover between two adjacent sec, default is 0<br>
returns data, label and indices for train, validation and test data sets as pydict<br>
</body>


## standardize
<body>
############  STANDARDIZE  #############<br>
data : pandas series, dataframe, list or numpy ndarray, input data to be standardized<br>
zero_std : float, value used to replace std values in case std is 0 for any column<br>
</body>
