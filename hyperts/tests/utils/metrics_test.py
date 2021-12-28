import numpy as np
import pandas as pd
from hyperts.utils.metrics import mse, rmse, mae, mape, smape

class Test_Metrics():

    def get_float_data(self):
        np_x1 = np.random.rand(50, 3)
        np_x2 = np.random.rand(50, 3)
        np_x3 = np.random.rand(50, 3)

        row_idx = np.random.choice(np.arange(len(np_x3)), size=5)
        col_idx = np.array([0, 2, 0, 1, 1])
        np_x3[(row_idx, col_idx)] = np.nan

        df_x1 = pd.DataFrame(np_x1, columns=['v1', 'v2', 'v3'])
        df_x2 = pd.DataFrame(np_x2, columns=['w1', 'w2', 'w3'])
        df_x3 = pd.DataFrame(np_x3, columns=['z1', 'z2', 'z3'])

        return np_x1, np_x2, np_x3, df_x1, df_x2, df_x3

    def forecast_score(self, func, np_x1, np_x2, np_x3, df_x1, df_x2, df_x3):
        assert func(np_x1, np_x2) > 0
        assert func(df_x1, np_x2) > 0
        assert func(np_x1, np_x3) > 0
        assert func(np_x1, df_x2) > 0
        assert func(np_x1, df_x3) > 0
        return True

    def test_forecast_regression_metrics(self):
        np_x1, np_x2, np_x3, df_x1, df_x2, df_x3 = self.get_float_data()

        assert self.forecast_score(mse, np_x1, np_x2, np_x3, df_x1, df_x2, df_x3)
        assert self.forecast_score(rmse, np_x1, np_x2, np_x3, df_x1, df_x2, df_x3)
        assert self.forecast_score(mae, np_x1, np_x2, np_x3, df_x1, df_x2, df_x3)
        assert self.forecast_score(mape, np_x1, np_x2, np_x3, df_x1, df_x2, df_x3)
        assert self.forecast_score(smape, np_x1, np_x2, np_x3, df_x1, df_x2, df_x3)
