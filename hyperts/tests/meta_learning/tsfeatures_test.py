import pandas as pd

from hyperts.datasets import load_random_univariate_forecast_dataset
from hyperts.framework.meta_learning import metafeatures_from_timeseries, normalization


class Test_TSMetaFeature():

    def test_forecast_metafeatures(self):
        df = load_random_univariate_forecast_dataset()
        df.drop(columns=['id'], inplace=True)
        metafeatures = metafeatures_from_timeseries(df, timestamp='ds', scale_ts=True)

        normalized_mf = normalization(metafeatures)

        assert isinstance(metafeatures, pd.DataFrame)
        assert isinstance(normalized_mf, pd.DataFrame)