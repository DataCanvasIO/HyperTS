import pytest

try:
    import tensorflow
    is_tf_installed = True
except:
    is_tf_installed = False

try:
    try:
        from prophet import Prophet
    except:
        from fbprophet import Prophet
    is_prophet_installed = True
except:
    is_prophet_installed = False

try:
    import sktime
    is_sktime_installed = True
except:
    is_sktime_installed = False

skip_if_not_tf = pytest.mark.skipif(not is_tf_installed, reason='Not install tensorflow environment.')
skip_if_not_prophet = pytest.mark.skipif(not is_prophet_installed, reason='Not install prophet environment.')
skip_if_not_sktime = pytest.mark.skipif(not is_sktime_installed, reason='Not install sktime environment.')