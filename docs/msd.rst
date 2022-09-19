msd
===

This module provides many useful functions and classes to ease works for a data scientist/ML engineer. The key elements are like below.

   -Data visualization
      - Time series data visualization\n
      - Drawing heatmap, table etc.
      - Drawing FIR filter response and filter effects (before and after applying filter) etc.
      - GRID plot for understanding relationship among features
      - Visualizing Spectrogram

   -Data processing
      - FIR filtering for time series data
      - Computing Spectogram
      - Data standardization and normalization
      - One hot encoding
      - Moving slope and mode for time series
   
   -Tools for ML
      - Feature ranking using simple statistics
      - Prediction evaluation for both classification (2 and more than 2 classes) and regression
      - Hyper-parameter optimization
      - splitting data set into train, validation and test using\n
         - Random splitting method
         - K-fold cross validation splitting
         - Sequence splitting (necessary for LSTM modeling)
      

.. automodule:: msdlib.msd
   :members:
   :undoc-members:
   :show-inheritance:

.. toctree::
   :maxdepth: 2

   msd.vis
   msd.processing