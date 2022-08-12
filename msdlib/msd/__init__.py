"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""


from .vis import (
    ProgressBar,
    word_length_error,
    name_separation,
    input_variable_error,
    plot_time_series,
    plot_heatmap,
    data_gridplot,
    plot_diag,
    get_color_from_cmap,
    get_named_colors,
    plot_table,
    plot_class_score,
    plot_regression_score
)

from .processing import (
    get_time_estimation,
    Filters,
    get_spectrogram,
    invalid_bins,
    grouped_mode,
    get_category_edges,
    get_edges_from_ts,
    each_row_max,
    moving_slope,
    standardize,
    normalize,
    feature_evaluator,
    get_weighted_scores,
    class_result,
    rsquare_rmse,
    regression_result,
    one_hot_encoding,
    SplitDataset,
    paramOptimizer
)


__all__ = [
    "ProgressBar",
    "word_length_error",
    "name_separation",
    "input_variable_error",
    "plot_time_series",
    "plot_heatmap",
    "data_gridplot",
    "plot_diag",
    "get_color_from_cmap",
    "get_named_colors",
    "plot_table",
    "plot_class_score",
    "plot_regression_score",
    "get_time_estimation",
    "Filters",
    "get_spectrogram",
    "invalid_bins",
    "grouped_mode",
    "get_category_edges",
    "get_edges_from_ts",
    "each_row_max",
    "moving_slope",
    "standardize",
    "normalize",
    "feature_evaluator",
    "get_weighted_scores",
    "class_result",
    "rsquare_rmse",
    "regression_result",
    "one_hot_encoding",
    "SplitDataset",
    "paramOptimizer"
]