import numpy as np
import pandas as pd
from hyperts.utils import get_tool_box

def from_3d_array_to_nested_df(data: np.ndarray,
                               columns: str = None,
                               cells_as_array: bool = False):
    """Convert Numpy ndarray with shape (nb_samples, series_length, nb_variables)
    into nested pandas DataFrame (with time series as numpy array or pandas Series in cells)

    Parameters
    ----------
    data : np.ndarray
        3-dimensional Numpy array to convert to nested pandas DataFrame format
    columns: list-like, default = None
        Optional list of names to use for naming nested DataFrame's columns
    cells_as_array : bool, default = False
        If True, then nested cells contain Numpy array
        If False, then nested cells contain pandas Series

    Returns
    ----------
    df : pd.DataFrame
    """

    tb = get_tool_box(data)
    return tb.from_3d_array_to_nested_df(data, columns, cells_as_array)


def from_nested_df_to_3d_array(data: pd.DataFrame):
    """Convert nested pandas DataFrame (with time series as numpy array or pandas Series in cells)
    into Numpy ndarray with shape (nb_samples, series_length, nb_variables).

    Parameters
    ----------
    data : pd.DataFrame
        Nested pandas DataFrame

    Returns
    -------
    data_3d : np.arrray
        3-dimensional NumPy array
    """

    tb = get_tool_box(data)
    return tb.from_nested_df_to_3d_array(data)


def random_train_test_split(*arrays,
                            test_size=None,
                            train_size=None,
                            random_state=None,
                            shuffle=True,
                            stratify=None):
    """Split arrays or matrices into random train and test subsets. This
    is a wrapper of scikit-learn's ``train_test_split`` that has shuffle.
    """
    tb = get_tool_box(arrays[0])
    return tb.random_train_test_split(*arrays,
                                      test_size=test_size,
                                      train_size=train_size,
                                      random_state=random_state,
                                      shuffle=shuffle,
                                      stratify=stratify)


def temporal_train_test_split(*arrays,
                              test_size=None,
                              train_size=None,
                              test_horizon=None):
    """Split arrays or matrices into sequential train and test subsets.This
    is a wrapper of scikit-learn's ``train_test_split`` that does not shuffle.

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0] Allowed inputs
    are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
    test_size : float, int or None, optional (default=None)
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25.
    train_size : float, int, or None, (default=None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    test_horizon: int or None, (default=None)
        If int, represents the forecast horizon length.
    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.
    """
    tb = get_tool_box(arrays[0])
    return tb.temporal_train_test_split(*arrays,
                                        test_size=test_size,
                                        test_horizon=test_horizon,
                                        train_size=train_size)
