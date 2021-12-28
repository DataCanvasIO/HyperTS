import dask.dataframe as dd
from hyperts.utils._base import get_tool_box
from hyperts.datasets import load_random_univariate_forecast_dataset

class Test_TSToolbox():

    def test_import_tstoolbox(self):
        X, y = load_random_univariate_forecast_dataset(return_X_y=True)
        X_dask = dd.from_pandas(X.head((10)), 1)
        tb = get_tool_box(X_dask)
        print(tb)
