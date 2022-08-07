"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""

# importing all necessary dependencies
import numpy as np
import pandas as pd
from scipy.io import wavfile
import matplotlib.pyplot as plt
import sounddevice as sd

import os
import sys
project_dir = os.getcwd()
sys.path.append(project_dir)
from msdlib import msd


savepath = 'examples/filters_and_spectrogram_example'
os.makedirs(savepath, exist_ok=True)

# loading example wav file
fs, data = wavfile.read('examples/data/example_tone_001.wav')
print('fs=', fs)
# converting stereo audio data into pandas dataframe
data = pd.DataFrame(data, columns=['stereo_1', 'stereo_2'])
data.index /= fs

# playing the sound (a sound of whistle recorded by myself)
sd.play(data['stereo_1'], fs)
sd.wait()

# plotting the time series data
fig, ax = plt.subplots(figsize=(30, 3))
ax.plot(data['stereo_1'], color='darkcyan')
ax.set_title('recorded_signal')
fig.tight_layout()
fig.savefig(os.path.join(savepath, 'recorded_signal.jpg'), bbox_inches='tight')
plt.show()

# Filters() example
key = 'stereo_1'

# filter definition
filt = msd.Filters(1/fs, N=1000, save=True,
                   savepath=savepath, show=True)
# visualizing frequency domain spectrum of the time series
filt.vis_spectrum(data[key], f_lim=[0, 8000], see_neg=False, figsize=(30, 3))

# applying band pass filter of cut offs 1800 - 2500 Hz and visualizing filter's spectrum
y = filt.apply(data[key], filt_type='bp', f_cut=[1800, 2500],
               order=10, response=True, plot=True, f_lim=[0, 8000])

# try other two to play around with other features

# # applying band pass filter of cut offs 1800 - 2500 Hz and visualizing the filtered signal's spectrum and time series plot
# y = filt.apply(data[key], filt_type='bp', f_cut=[1800, 2500],
#                order=10, response=False, plot=True, f_lim=[0, 8000])

# # applying band pass filter of cut offs 1800 - 2500 Hz and visualizing all together
# y = filt.apply(data[key], filt_type='bp', f_cut=[1800, 2500],
#                order=10, response=True, plot=False, f_lim=[0, 8000])

# playing filtered sound
sd.play(y.values, fs)
sd.wait()

# putting spectrogram data inside a variable and also showing spectrogram
# visualizing only upto frequency of 1/8 th of nyquest frequency by using vis_freq for clear visualization
sxx = msd.get_spectrogram(data['stereo_1'], fs=fs, nperseg=int(fs/2), noverlap=int(fs/5), mode='psd',
                          vis_frac=1/8, ret_sxx=True, show=False, save=True, savepath=savepath)
print(sxx)

# spectrogram after applying filter
sxx = msd.get_spectrogram(y, fs=fs, nperseg=int(fs/2), noverlap=int(fs/5), mode='psd',
                          vis_frac=1/8, ret_sxx=True, show=False, save=True, savepath=savepath)
