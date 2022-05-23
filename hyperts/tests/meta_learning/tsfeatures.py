import pandas as pd

from hyperts.datasets import load_random_univariate_forecast_dataset
from hyperts.framework.meta_learning.tsfeatures import metafeatures_from_timeseries


class Test_TSMetaFeature:

    def test_forecast_metafeatures(self):
        df = load_random_univariate_forecast_dataset()
        df.drop(columns=['id'], inplace=True)
        metafeatures = metafeatures_from_timeseries(df, timestamp='ds')

        assert isinstance(metafeatures, pd.DataFrame)