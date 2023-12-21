# -*- coding: utf-8 -*-

import pandas as pd

__all__ = [
    "MTYPE_REGISTER_PANEL",
    "MTYPE_LIST_PANEL",
    "MTYPE_REGISTER_SERIES",
    "MTYPE_LIST_SERIES",
    "MTYPE_REGISTER_TABLE",
    "MTYPE_LIST_TABLE",
    "MTYPE_REGISTER_ALIGNMENT",
    "MTYPE_LIST_ALIGNMENT",
]


MTYPE_REGISTER_PANEL = [
    (
        "nested_univ",
        "Panel",
        "pd.DataFrame with one column per variable, pd.Series in cells",
    ),
    (
        "numpy3D",
        "Panel",
        "3D np.array of format (n_instances, n_columns, n_timepoints)",
    ),
    (
        "numpyflat",
        "Panel",
        "2D np.array of format (n_instances, n_columns*n_timepoints)",
    ),
    ("pd-multiindex", "Panel", "pd.DataFrame with multi-index (instances, timepoints)"),
    ("pd-wide", "Panel", "pd.DataFrame in wide format, cols = (instance*timepoints)"),
    (
        "pd-long",
        "Panel",
        "pd.DataFrame in long format, cols = (index, time_index, column)",
    ),
    ("df-list", "Panel", "list of pd.DataFrame"),
]

MTYPE_REGISTER_SERIES = [
    ("pd.Series", "Series", "pd.Series representation of a univariate series"),
    (
        "pd.DataFrame",
        "Series",
        "pd.DataFrame representation of a uni- or multivariate series",
    ),
    (
        "np.ndarray",
        "Series",
        "2D numpy.ndarray with rows=samples, cols=variables, index=integers",
    ),
]

MTYPE_REGISTER_TABLE = [
    ("pd_DataFrame_Table", "Table", "pd.DataFrame representation of a data table"),
    ("numpy1D", "Table", "1D np.narray representation of a univariate table"),
    ("numpy2D", "Table", "2D np.narray representation of a univariate table"),
    ("pd_Series_Table", "Table", "pd.Series representation of a data table"),
    ("list_of_dict", "Table", "list of dictionaries with primitive entries"),
]

MTYPE_REGISTER_ALIGNMENT = [
    (
        "alignment",
        "Alignment",
        "pd.DataFrame in alignment format, values are iloc index references",
    ),
    (
        "alignment_loc",
        "Alignment",
        "pd.DataFrame in alignment format, values are loc index references",
    ),
]

MTYPE_REGISTER_HIERARCHICAL = [
    (
        "pd_multiindex_hier",
        "Hierarchical",
        "pd.DataFrame with MultiIndex",
    ),
]

MTYPE_LIST_PANEL = pd.DataFrame(MTYPE_REGISTER_PANEL)[0].values
MTYPE_LIST_SERIES = pd.DataFrame(MTYPE_REGISTER_SERIES)[0].values
MTYPE_LIST_TABLE = pd.DataFrame(MTYPE_REGISTER_TABLE)[0].values
MTYPE_LIST_ALIGNMENT = pd.DataFrame(MTYPE_REGISTER_ALIGNMENT)[0].values
MTYPE_LIST_HIERARCHICAL = pd.DataFrame(MTYPE_REGISTER_HIERARCHICAL)[0].values