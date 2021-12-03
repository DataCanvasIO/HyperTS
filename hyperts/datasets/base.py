import datetime
import numpy as np
import pandas as pd
from random import random
from os.path import join, dirname

def load_network_traffic(univariate=False):
    """Network Traffic Forecast

    """
    module_path = dirname(__file__)
    data_file_name = join(module_path, 'network_traffic_forecast.csv')
    df = pd.read_csv(data_file_name, encoding='utf-8')
    if univariate:
        variable = np.random.choice(['Var_1', 'Var_2', 'Var_3', 'Var_4', 'Var_5', 'Var_6'], size=1)[0]
        df = df[['TimeStamp', variable, 'HourSin', 'WeekCos', 'CBWD']]

    return df


def load_random_univariate_forecast_dataset():

    def get_num(num):
        return 0 if num < 0.5 else 1

    id_data = [get_num(random()) for i in range(100)]
    id_data[10] = None

    X = pd.DataFrame({'ds': pd.date_range("2013-01-01", periods=100), 'id': id_data})

    y = pd.DataFrame({'value':  np.random.rand(1, 100)[0].tolist()})
    return X, y


def load_random_multivariate_forecast_dataset():
    now_date = datetime.datetime.now()
    # contrived dataset with dependency
    data = list()
    X = []
    for i in range(100):
        now_date = now_date + datetime.timedelta(days=1)
        X.append(now_date)
        v1 = i + random()
        v2 = v1 + random()
        row = [v1, v2]
        data.append(row)
    X = pd.DataFrame(data={'ds': X})
    y = pd.DataFrame(data=data)
    y.columns = ['Var_1', 'Var_2']
    return X, y


if __name__ == '__main__':
    df = load_network_traffic()
    print('finish')