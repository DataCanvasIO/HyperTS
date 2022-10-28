import pandas as pd
import dask.dataframe as dd

from hyperts.utils._base import get_tool_box
from hyperts.datasets import load_random_univariate_forecast_dataset

class Test_TSToolbox():

    def test_import_tstoolbox(self):
        X, y = load_random_univariate_forecast_dataset(return_X_y=True)
        tb = get_tool_box(X, y)

        period = tb.fft_infer_period(y)
        assert isinstance(period, int)

        lags = tb.infer_window_size(max_size=100, freq='D')
        assert isinstance(lags, list)

        date_covariates = tb.generate_time_covariates(start_date='2022-01-01', periods=100, freq='H')
        assert isinstance(date_covariates, pd.DataFrame)

        X_dask = dd.from_pandas(X.head((10)), 1)
        tbdk = get_tool_box(X_dask)
        print(tbdk)
