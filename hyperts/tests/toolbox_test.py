import hyperts.utils.toolbox as tb
import dask.dataframe as dd
import hyperts.datasets.base as ds


def testimporttoolbox():
    print("test testImportToolBox")
    X, y = ds.load_random_univariate_forecast_dataset(return_X_y=True)
    X_dask = dd.from_pandas(X.head((10)), 1)
    tstb = tb.get_tool_box(X_dask)
    print(tstb)
