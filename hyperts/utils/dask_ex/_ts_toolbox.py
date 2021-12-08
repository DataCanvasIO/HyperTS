from ..toolbox import TSToolBox
import dask
import dask.dataframe as dd
import dask.array as da

class TSDaskToolBox(TSToolBox):
    acceptable_types = (dd.DataFrame, dd.Series, da.Array)
    compute = dask.compute