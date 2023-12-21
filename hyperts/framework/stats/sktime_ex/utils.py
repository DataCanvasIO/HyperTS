from typing import List, Union
import os
import numpy as np
from copy import deepcopy
from sklearn import clone
from sklearn.ensemble._base import _set_random_states
from hyperts.framework.stats.sktime_ex.datatypes._registry import MTYPE_REGISTER_PANEL
from hyperts.framework.stats.sktime_ex.datatypes._registry import MTYPE_REGISTER_HIERARCHICAL
from hyperts.framework.stats.sktime_ex.datatypes._registry import MTYPE_REGISTER_ALIGNMENT
from hyperts.framework.stats.sktime_ex.datatypes._registry import MTYPE_REGISTER_TABLE
from hyperts.framework.stats.sktime_ex.datatypes._registry import MTYPE_REGISTER_SERIES

MTYPE_REGISTER = []
MTYPE_REGISTER += MTYPE_REGISTER_SERIES
MTYPE_REGISTER += MTYPE_REGISTER_PANEL
MTYPE_REGISTER += MTYPE_REGISTER_HIERARCHICAL
MTYPE_REGISTER += MTYPE_REGISTER_ALIGNMENT
MTYPE_REGISTER += MTYPE_REGISTER_TABLE

from hyperts.framework.stats.sktime_ex.datatypes import check_dict_Alignment
from hyperts.framework.stats.sktime_ex.datatypes import check_dict_Series, convert_dict_Series
from hyperts.framework.stats.sktime_ex.datatypes import check_dict_Panel, convert_dict_Panel
from hyperts.framework.stats.sktime_ex.datatypes import check_dict_Table, convert_dict_Table
from hyperts.framework.stats.sktime_ex.datatypes import check_dict_Hierarchical, convert_dict_Hierarchical

check_dict = dict()
check_dict.update(check_dict_Series)
check_dict.update(check_dict_Panel)
check_dict.update(check_dict_Table)
check_dict.update(check_dict_Hierarchical)
check_dict.update(check_dict_Alignment)

convert_dict = dict()
convert_dict.update(convert_dict_Series)
convert_dict.update(convert_dict_Panel)
convert_dict.update(convert_dict_Table)
convert_dict.update(convert_dict_Hierarchical)


def _check_scitype_valid(scitype: str = None):
    """Check validity of scitype."""
    valid_scitypes = list(set([x[1] for x in check_dict.keys()]))

    if not isinstance(scitype, str):
        raise TypeError(f"scitype should be a str but found {type(scitype)}")

    if scitype is not None and scitype not in valid_scitypes:
        raise TypeError(scitype + " is not a supported scitype")

def _ret(valid, msg, metadata, return_metadata):
    if return_metadata:
        return valid, msg, metadata
    else:
        return valid

def _coerce_list_of_str(obj, var_name="obj"):
    """Check whether object is string or list of string.

    Parameters
    ----------
    obj - object to check
    var_name: str, optional, default="obj" - name of input in error messages

    Returns
    -------
    list of str
        equal to obj if was a list; equal to [obj] if obj was a str
        note: if obj was a list, return is not a copy, but identical

    Raises
    ------
    TypeError if obj is not a str or list of str
    """
    if isinstance(obj, str):
        obj = [obj]
    elif isinstance(obj, list):
        if not np.all([isinstance(x, str) for x in obj]):
            raise TypeError(f"{var_name} must be a string or list of strings")
    else:
        raise TypeError(f"{var_name} must be a string or list of strings")

    return obj

def _clone_estimator(base_estimator, random_state=None):
    estimator = clone(base_estimator)

    if random_state is not None:
        _set_random_states(estimator, random_state)

    return estimator

def _slope(y, axis=0):
    """Find the slope for each series of y.

    Parameters
    ----------
    y: np.ndarray
        Time series
    axis : int, optional (default=0)
        Axis along which to compute slope

    Returns
    -------
    slope : np.ndarray
        Time series slope
    """
    # Make sure y is always at least 2-dimensional
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Generate time index with correct shape for broadcasting
    shape = np.ones(y.ndim, dtype=int)
    shape[axis] *= -1
    x = np.arange(y.shape[axis]).reshape(shape) + 1

    # Precompute mean
    x_mean = x.mean()

    # Compute slope along given axis
    return (np.mean(y * x, axis=axis) - x_mean * np.mean(y, axis=axis)) / (
        (x * x).mean() - x_mean ** 2
    )

def is_int(x) -> bool:
    """Check if x is of integer type, but not boolean."""
    # boolean are subclasses of integers in Python, so explicitly exclude them
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)

def mtype_to_scitype(mtype: str, return_unique=False):
    """Infer scitype belonging to mtype.

    Parameters
    ----------
    mtype : str, or list of str, or nested list/str object, or None
        mtype(s) to find scitype of

    Returns
    -------
    scitype : str, or list of str, or nested list/str object, or None
        if str, returns scitype belonging to mtype, if mtype is str
        if list, returns this function element-wise applied
        if nested list/str object, replaces mtype str by scitype str
        if None, returns None
    return_unique : bool, default=False
        if True, makes

    Raises
    ------
    TypeError, if input is not of the type specified
    ValueError, if there are two scitypes for the/some mtype string
        (this should not happen in general, it means there is a bug)
    ValueError, if there is no scitype for the/some mtype string
    """
    # handle the "None" case first
    if mtype is None or mtype == "None":
        return None
    # recurse if mtype is a list
    if isinstance(mtype, list):
        scitype_list = [mtype_to_scitype(x) for x in mtype]
        if return_unique:
            scitype_list = list(set(scitype_list))
        return scitype_list

    # checking for type. Checking str is enough, recursion above will do the rest.
    if not isinstance(mtype, str):
        raise TypeError(
            "mtype must be str, or list of str, nested list/str object, or None"
        )

    scitype = [k[1] for k in MTYPE_REGISTER if k[0] == mtype]

    if len(scitype) > 1:
        raise ValueError("multiple scitypes match the mtype, specify scitype")

    if len(scitype) < 1:
        raise ValueError(f"{mtype} is not a supported mtype")

    return scitype[0]

def check_is_mtype(
    obj,
    mtype: Union[str, List[str]],
    scitype: str = None,
    return_metadata=False,
    var_name="obj",
):
    """Check object for compliance with mtype specification, return metadata.

    Parameters
    ----------
    obj - object to check
    mtype: str or list of str, mtype to check obj as
        valid mtype strings are in datatypes.MTYPE_REGISTER (1st column)
    scitype: str, optional, scitype to check obj as; default = inferred from mtype
        if inferred from mtype, list elements of mtype need not have same scitype
        valid mtype strings are in datatypes.SCITYPE_REGISTER (1st column)
    return_metadata - bool, optional, default=False
        if False, returns only "valid" return
        if True, returns all three return objects
    var_name: str, optional, default="obj" - name of input in error messages

    Returns
    -------
    valid: bool - whether obj is a valid object of mtype/scitype
    msg: str or list of str - error messages if object is not valid, otherwise None
            str if mtype is str; list of len(mtype) with message per mtype if list
            returned only if return_metadata is True
    metadata: dict - metadata about obj if valid, otherwise None
            returned only if return_metadata is True
        Keys populated depend on (assumed, otherwise identified) scitype of obj.
        Always returned:
            "mtype": str, mtype of obj (assumed or inferred)
            "scitype": str, scitype of obj (assumed or inferred)
        For scitype "Series":
            "is_univariate": bool, True iff series has one variable
            "is_equally_spaced": bool, True iff series index is equally spaced
            "is_empty": bool, True iff series has no variables or no instances
            "has_nans": bool, True iff the series contains NaN values
        For scitype "Panel":
            "is_univariate": bool, True iff all series in panel have one variable
            "is_equally_spaced": bool, True iff all series indices are equally spaced
            "is_equal_length": bool, True iff all series in panel are of equal length
            "is_empty": bool, True iff one or more of the series in the panel are empty
            "is_one_series": bool, True iff there is only one series in the panel
            "has_nans": bool, True iff the panel contains NaN values
            "n_instances": int, number of instances in the panel
        For scitype "Table":
            "is_univariate": bool, True iff table has one variable
            "is_empty": bool, True iff table has no variables or no instances
            "has_nans": bool, True iff the panel contains NaN values
        For scitype "Alignment":
            currently none

    Raises
    ------
    TypeError if no checks defined for mtype/scitype combination
    TypeError if mtype input argument is not of expected type
    """
    mtype = _coerce_list_of_str(mtype, var_name="mtype")

    valid_keys = check_dict.keys()

    # we loop through individual mtypes in mtype and see whether they pass the check
    #  for each check we remember whether it passed and what it returned
    msg = []
    found_mtype = []
    found_scitype = []

    for m in mtype:
        if scitype is None:
            scitype_of_m = mtype_to_scitype(m)
        else:
            _check_scitype_valid(scitype)
            scitype_of_m = scitype
        key = (m, scitype_of_m)
        if (m, scitype_of_m) not in valid_keys:
            raise TypeError(f"no check defined for mtype {m}, scitype {scitype_of_m}")

        res = check_dict[key](obj, return_metadata=return_metadata, var_name=var_name)

        if return_metadata:
            check_passed = res[0]
        else:
            check_passed = res

        if check_passed:
            found_mtype.append(m)
            found_scitype.append(scitype_of_m)
            final_result = res
        elif return_metadata:
            msg.append(res[1])

    # there are three options on the result of check_is_mtype:
    # a. two or more mtypes are found - this is unexpected and an error with checks
    if len(found_mtype) > 1:
        raise TypeError(
            f"Error in check_is_mtype, more than one mtype identified: {found_mtype}"
        )
    # b. one mtype is found - then return that mtype
    elif len(found_mtype) == 1:
        if return_metadata:
            # add the mtype return to the metadata
            final_result[2]["mtype"] = found_mtype[0]
            final_result[2]["scitype"] = found_scitype[0]
            # final_result already has right shape and dependency on return_metadata
            return final_result
        else:
            return True
    # c. no mtype is found - then return False and all error messages if requested
    else:
        if len(msg) == 1:
            msg = msg[0]

        return _ret(False, msg, None, return_metadata)

def infer_mtype(obj, as_scitype: Union[str, List[str]] = None):
    """Infer the mtype of an object considered as a specific scitype.

    Parameters
    ----------
    obj : object to infer type of - any type, should comply with and mtype spec
        if as_scitype is provided, this needs to be mtype belonging to scitype
    as_scitype : str, list of str, or None, optional, default=None
        name of scitype(s) the object "obj" is considered as, finds mtype for that
        if None (default), does not assume a specific as_scitype and tests all mtypes
            generally, as_scitype should be provided for maximum efficiency
        valid scitype type strings are in datatypes.SCITYPE_REGISTER (1st column)

    Returns
    -------
    str - the inferred mtype of "obj", a valid mtype string
            or None, if obj is None
        mtype strings with explanation are in datatypes.MTYPE_REGISTER

    Raises
    ------
    TypeError if no type can be identified, or more than one type is identified
    """
    if obj is None:
        return None

    if as_scitype is not None:
        as_scitype = _coerce_list_of_str(as_scitype, var_name="as_scitype")
        for scitype in as_scitype:
            _check_scitype_valid(scitype)

    if as_scitype is None:
        m_plus_scitypes = [(x[0], x[1]) for x in check_dict.keys()]
    else:
        m_plus_scitypes = [
            (x[0], x[1]) for x in check_dict.keys() if x[1] in as_scitype
        ]

    res = [
        m_plus_scitype[0]
        for m_plus_scitype in m_plus_scitypes
        if check_is_mtype(obj, mtype=m_plus_scitype[0], scitype=m_plus_scitype[1])
    ]

    if len(res) > 1:
        raise TypeError(
            f"Error in check_is_mtype, more than one mtype identified: {res}"
        )

    if len(res) < 1:
        raise TypeError("No valid mtype could be identified")

    return res[0]

def check_is_scitype(
    obj,
    scitype: Union[str, List[str]],
    return_metadata=False,
    var_name="obj",
):
    """Check object for compliance with mtype specification, return metadata.

    Parameters
    ----------
    obj - object to check
    scitype: str or list of str, scitype to check obj as
        valid mtype strings are in datatypes.SCITYPE_REGISTER
    return_metadata - bool, optional, default=False
        if False, returns only "valid" return
        if True, returns all three return objects
    var_name: str, optional, default="obj" - name of input in error messages

    Returns
    -------
    valid: bool - whether obj is a valid object of mtype/scitype
    msg: str or list of str - error messages if object is not valid, otherwise None
            str if mtype is str; list of len(mtype) with message per mtype if list
            returned only if return_metadata is True
    metadata: dict - metadata about obj if valid, otherwise None
            returned only if return_metadata is True
        Fields depend on scitpe.
        Always returned:
            "mtype": str, mtype of obj (assumed or inferred)
                mtype strings with explanation are in datatypes.MTYPE_REGISTER
            "scitype": str, scitype of obj (assumed or inferred)
                scitype strings with explanation are in datatypes.SCITYPE_REGISTER
        For scitype "Series":
            "is_univariate": bool, True iff series has one variable
            "is_equally_spaced": bool, True iff series index is equally spaced
            "is_empty": bool, True iff series has no variables or no instances
            "has_nans": bool, True iff the series contains NaN values
        For scitype "Panel":
            "is_univariate": bool, True iff all series in panel have one variable
            "is_equally_spaced": bool, True iff all series indices are equally spaced
            "is_equal_length": bool, True iff all series in panel are of equal length
            "is_empty": bool, True iff one or more of the series in the panel are empty
            "is_one_series": bool, True iff there is only one series in the panel
            "has_nans": bool, True iff the panel contains NaN values
            "n_instances": int, number of instances in the panel
        For scitype "Table":
            "is_univariate": bool, True iff table has one variable
            "is_empty": bool, True iff table has no variables or no instances
            "has_nans": bool, True iff the panel contains NaN values
        For scitype "Alignment":
            currently none
    Raises
    ------
    TypeError if scitype input argument is not of expected type
    """
    scitype = _coerce_list_of_str(scitype, var_name="scitype")

    for x in scitype:
        _check_scitype_valid(x)

    valid_keys = check_dict.keys()

    # find all the mtype keys corresponding to the scitypes
    keys = [x for x in valid_keys if x[1] in scitype]

    # storing the msg retursn
    msg = []
    found_mtype = []
    found_scitype = []

    for key in keys:
        res = check_dict[key](obj, return_metadata=return_metadata, var_name=var_name)

        if return_metadata:
            check_passed = res[0]
        else:
            check_passed = res

        if check_passed:
            final_result = res
            found_mtype.append(key[0])
            found_scitype.append(key[1])
        elif return_metadata:
            msg.append(res[1])

    # there are three options on the result of check_is_mtype:
    # a. two or more mtypes are found - this is unexpected and an error with checks
    if len(found_mtype) > 1:
        raise TypeError(
            f"Error in check_is_mtype, more than one mtype identified: {found_mtype}"
        )
    # b. one mtype is found - then return that mtype
    elif len(found_mtype) == 1:
        if return_metadata:
            # add the mtype return to the metadata
            final_result[2]["mtype"] = found_mtype[0]
            # add the scitype return to the metadata
            final_result[2]["scitype"] = found_scitype[0]
            # final_result already has right shape and dependency on return_metadata
            return final_result
        else:
            return True
    # c. no mtype is found - then return False and all error messages if requested
    else:
        if len(msg) == 1:
            msg = msg[0]

        return _ret(False, msg, None, return_metadata)

def convert(
    obj,
    from_type: str,
    to_type: str,
    as_scitype: str = None,
    store=None,
    store_behaviour: str = None,
):
    """Convert objects between different machine representations, subject to scitype.

    Parameters
    ----------
    obj : object to convert - any type, should comply with mtype spec for as_scitype
    from_type : str - the type to convert "obj" to, a valid mtype string
    to_type : str - the type to convert "obj" to, a valid mtype string
    as_scitype : str, optional - name of scitype the object "obj" is considered as
        default = inferred from from_type
    store : optional, reference of storage for lossy conversions, default=None (no ref)
        is updated by side effect if not None and store_behaviour="reset" or "update"
    store_behaviour : str, optional, one of None (default), "reset", "freeze", "update"
        "reset" - store is emptied and then updated from conversion
        "freeze" - store is read-only, may be read/used by conversion but not changed
        "update" - store is updated from conversion and retains previous contents
        None - automatic: "update" if store is empty and not None; "freeze", otherwise

    Returns
    -------
    converted_obj : to_type - object obj converted to to_type
                    if obj was None, returns None

    Raises
    ------
    KeyError if conversion is not implemented
    TypeError or ValueError if inputs do not match specification
    """
    if obj is None:
        return None

    # input type checks
    if not isinstance(to_type, str):
        raise TypeError("to_type must be a str")
    if not isinstance(from_type, str):
        raise TypeError("from_type must be a str")
    if as_scitype is None:
        as_scitype = mtype_to_scitype(to_type)
    elif not isinstance(as_scitype, str):
        raise TypeError("as_scitype must be str or None")
    if store is not None and not isinstance(store, dict):
        raise TypeError("store must be a dict or None")
    if store_behaviour not in [None, "reset", "freeze", "update"]:
        raise ValueError(
            'store_behaviour must be one of "reset", "freeze", "update", or None'
        )
    if store_behaviour is None and store == {}:
        store_behaviour = "update"
    if store_behaviour is None and store != {}:
        store_behaviour = "freeze"

    key = (from_type, to_type, as_scitype)

    if key not in convert_dict.keys():
        raise NotImplementedError(
            "no conversion defined from type " + str(from_type) + " to " + str(to_type)
        )

    if store_behaviour == "freeze":
        store = deepcopy(store)
    elif store_behaviour == "reset":
        # note: this is a side effect on store
        store.clear()
    elif store_behaviour == "update":
        # store is passed to convert_obj by reference, unchanged
        # this "elif" is here for clarity, to cover all three values
        pass
    else:
        raise RuntimeError(
            "bug: unrechable condition error, store_behaviour has unexpected value"
        )

    converted_obj = convert_dict[key](obj, store=store)

    return converted_obj

# conversion based on queriable type to specified target
def convert_to(
    obj,
    to_type: str,
    as_scitype: str = None,
    store=None,
    store_behaviour: str = None,
):
    """Convert object to a different machine representation, subject to scitype.

    Parameters
    ----------
    obj : object to convert - any type, should comply with mtype spec for as_scitype
    to_type : str - the type to convert "obj" to, a valid mtype string
              or list - admissible types for conversion to
    as_scitype : str, optional - name of scitype the object "obj" is considered as
        default = inferred from mtype of obj, which is in turn inferred internally
    store : reference of storage for lossy conversions, default=None (no store)
        is updated by side effect if not None and store_behaviour="reset" or "update"
    store_behaviour : str, optional, one of None (default), "reset", "freeze", "update"
        "reset" - store is emptied and then updated from conversion
        "freeze" - store is read-only, may be read/used by conversion but not changed
        "update" - store is updated from conversion and retains previous contents
        None - automatic: "update" if store is empty and not None; "freeze", otherwise

    Returns
    -------
    converted_obj : to_type - object obj converted to to_type, if to_type is str
                     if to_type is list, converted to to_type[0],
                        unless from_type in to_type, in this case converted_obj=obj
                    if obj was None, returns None

    Raises
    ------
    TypeError if machine type of input "obj" is not recognized
    KeyError if conversion is not implemented
    TypeError or ValueError if inputs do not match specification
    """
    if obj is None:
        return None

    if isinstance(to_type, list):
        if not np.all(isinstance(x, str) for x in to_type):
            raise TypeError("to_type must be a str or list of str")
    elif not isinstance(to_type, str):
        raise TypeError("to_type must be a str or list of str")

    if as_scitype is None:
        if isinstance(to_type, str):
            as_scitype = mtype_to_scitype(to_type)
        else:
            as_scitype = mtype_to_scitype(to_type[0])
    elif not isinstance(as_scitype, str):
        raise TypeError("as_scitype must be a str or None")

    from_type = infer_mtype(obj=obj, as_scitype=as_scitype)

    # if to_type is a list:
    if isinstance(to_type, list):
        # no conversion of from_type is in the list
        if from_type in to_type:
            to_type = from_type
        # otherwise convert to first element
        else:
            to_type = to_type[0]

    converted_obj = convert(
        obj=obj,
        from_type=from_type,
        to_type=to_type,
        as_scitype=as_scitype,
        store=store,
        store_behaviour=store_behaviour,
    )

    return converted_obj


def check_n_jobs(n_jobs: int) -> int:
    """Check `n_jobs` parameter according to the scikit-learn convention.

    Parameters
    ----------
    n_jobs : int, positive or -1
        The number of jobs for parallelization.

    Returns
    -------
    n_jobs : int
        Checked number of jobs.
    """
    # scikit-learn convention
    # https://scikit-learn.org/stable/glossary.html#term-n-jobs
    if n_jobs is None:
        return 1
    elif not is_int(n_jobs):
        raise ValueError(f"`n_jobs` must be None or an integer, but found: {n_jobs}")
    elif n_jobs < 0:
        return os.cpu_count() - n_jobs + 1
    else:
        return n_jobs