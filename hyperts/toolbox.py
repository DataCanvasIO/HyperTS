import collections

import numpy as np
import pandas as pd
from hyperts.utils import get_tool_box
from hyperts.framework.meta_learning import tsfeatures
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils.validation import check_random_state
from sklearn.utils.random import sample_without_replacement

from hyperts.utils.tstoolbox import TSToolBox


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


def metafeatures_from_timeseries(
        x : pd.DataFrame,
        timestamp : str,
        period=None,
        scale_ts=True,
        freq_mapping_dict=None,
        features_list=None):
    """
    Extracting the meta-features of time series.

    Parameters
    ----------
    x: pd.DataFrame, the time series.
    timestamp: str, timestamp name of x.
    period: int or None, the seasonal of the time series, default None.
    scale_ts: bool, whether scale original time series.
    freq_mapping_dict, dict, default {'H': 24, 'D': 7, 'W': 54, 'M': 12,
        'Q': 4, 'Y': 1, 'A': 1, 'S': 60, 'T': 60}.
    features_list, List[str], default ['simple', 'all'].
    """
    return tsfeatures.metafeatures_from_timeseries(x,
                                                   timestamp,
                                                   period=period,
                                                   scale_ts=scale_ts,
                                                   freq_mapping_dict=freq_mapping_dict,
                                                   features_list=features_list)


def generate_anomaly_pseudo_ground_truth(
        X_train,
        X_test=None,
        local_region_size: int=30,
        local_max_features: float=1.0,
        local_region_iterations: int=20,
        generate_train_label_type: str='iforest',
        contamination: float=0.05,
        random_state=None):
    """Genrate pseudo ground truth for anomaly detection.

    Parameters
    ----------
    X_train : numpy array of shape (n_samples, n_features).
    X_test : numpy array of shape (n_samples, n_features).
    local_region_size : int, optional (default=30)
        Number of training points to consider in each iteration of the local
        region generation process (30 by default).
    local_max_features : float in (0.5, 1.), optional (default=1.0)
        Maximum proportion of number of features to consider when defining the
        local region (1.0 by default).
    local_region_iterations : int, optional (default=20)
        Number of iteration of the local region generation process.
    generate_train_label_type : str, optional (default='iforest')
        The method of genetating training pseudo labels.
    contamination : 'auto' or float, optional (default=0.05)
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the scores of the samples.
    random_state : RandomState, optional (default=None)
        A random number generator instance to define the state of the random
        permutations generator.

    References
    ----------

    """
    if not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train)

    if len(X_train.shape) == 1:
        X_train = X_train.reshape(-1, 1)

    check_array(X_train)
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)

    # Generate train pseudo ground truth process
    if generate_train_label_type == 'iforest':
        from sklearn.ensemble import IsolationForest
        detector = IsolationForest(contamination=contamination)
        detector.fit(X_train_norm)
        decision_func = detector.decision_function(X_train_norm)
        train_pseudo_labels = np.zeros_like(decision_func, dtype=int)
        train_pseudo_labels[decision_func < 0] = 1
    else:
        raise ValueError(f'This type is not spported.')

    if X_test is None:
        return train_pseudo_labels.reshape(-1, 1), None
    else:
        if not isinstance(X_test, np.ndarray):
            X_test = np.array(X_test)

    # Generate test pseudo ground truth process
    check_array(X_test)
    random_state = check_random_state(random_state)
    local_region_list = [[]] * X_test.shape[0]
    final_local_region_list = [[]] * X_test.shape[0]
    local_region_threshold = int(local_region_iterations / 2)

    if len(X_test.shape) == 1:
        X_test = X_test.reshape(-1, 1)

    n_features = X_train.shape[1]

    if local_max_features > 1.0:
        local_max_features = 1.0

    local_min_features = 0.5
    if n_features * local_min_features < 1:
        local_min_features = 1.0

    X_test_norm = scaler.transform(X_test)

    min_features = n_features * local_min_features
    max_features = n_features * local_max_features
    for _ in range(local_region_iterations):
        if local_min_features == local_max_features:
            feature_indices = range(0, n_features)
        else:
            random_n_features = random_state.randint(min_features, max_features)
            feature_indices = sample_without_replacement(n_population=n_features,
                          n_samples=random_n_features, random_state=random_state)

        # Bulid KDTree out of training subspace
        tree = KDTree(X_train_norm[:, feature_indices])

        # Find neighbors of each test instance
        _, index_arr = tree.query(X_test_norm[:, feature_indices], k=local_region_size)

        # Add neighbors to local region list
        for i in range(X_test_norm.shape[0]):
            local_region_list[i]  = local_region_list[i] + index_arr[i, :].tolist()

    # Keep nearby points which occur at least local_region_threshold times
    for j in range(X_test_norm.shape[0]):
        tmp = []
        for item, count in collections.Counter(local_region_list[j]).items():
            if count > local_region_threshold:
                tmp.append(item)
        decrease_value = 0
        while len(tmp) < 2:
            decrease_value += 1
            assert decrease_value < local_region_threshold
            tmp = []
            for item, count in collections.Counter(local_region_list[j]).items():
                if count > local_region_threshold - decrease_value:
                    tmp.append(item)
        final_local_region_list[j] = tmp

    # Generate test pseudo ground truth
    test_pseudo_labels = np.zeros(shape=(X_test.shape[0], 1), dtype=int)
    for k in range(X_test_norm.shape[0]):
        train_local_pseudo_labels = train_pseudo_labels[final_local_region_list[k]]
        if np.sum(train_local_pseudo_labels) > 1:
            test_pseudo_labels[k] = 1

    train_pseudo_labels = train_pseudo_labels.reshape(-1, 1)
    test_pseudo_labels = test_pseudo_labels.reshape(-1, 1)

    return train_pseudo_labels, test_pseudo_labels