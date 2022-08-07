.. msdlib documentation master file, created by
   sphinx-quickstart on Mon Sep  6 21:39:32 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


========================
Motivation behind msdlib
========================

.. image:: ../msdlib_logo_fit.png
   :width: 20%

msdlib is a collection of frequently used functions and modules. A common data scientist or data analyst has to do a lot of things. He has to work on data processing, data cleaning, data visualization, story-telling, insight generation, trend analysis etc.

Additionally, he aslo needs to take care algorithm development for data modeling. Now-a-days Machine Learning and Deep Learning are very efficient techniques to model data and provide solutions like binary and multi-class classification, regression etc. Additionally, some programmers need to build much more complex models too.

The idea of this library comes to light from this concept to make every general tasks easier. To prepare business level plots, tables and heatmaps very easily, to build Deep LEarning models very fast and train them with minimal coding.

msdlib provides supports to different types of things altogether. Some of the core features are-

- Generalized time series data visualization library for multivariate time series data
- Quick heatmap and table generation functions
- dataset splitting method (random, k-fold cross-validation and sequece splitting for LSTM)
- easy hyper-parameter optimization module
- Scikit-like wrapper for Pytorch models. (fit(), predict() and evaluate() for regression, binary classification and multi-label classification)
- Hyper-parameter optimization of Pytorch Deep Learning model.
- regression and classification result calculation functionality with many metrics (precision, recall, f1 score, accuracy, specificity, r-square, rmse etc.)
- Statistical feature importance functionality
- FIR filtering and filter and data visualization functionality with minimal coding.

msdlib documentation
====================

Get started from this `page <README.html>`_

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   README
   msd
   mlutils
   msdbacktest
   dataset
   msdExceptions
   stats


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
