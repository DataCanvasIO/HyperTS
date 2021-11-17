# https://facebook.github.io/prophet/docs/quick_start.html
import pandas as pd
from fbprophet import Prophet
from fbprophet import __version__
print(__version__)

df = pd.read_csv('C:/Users/wuhf/OpenSource/prophet/examples/example_wp_log_peyton_manning.csv')
df.head()

m = Prophet(interval_width=0.6)
m.fit(df)


future = m.make_future_dataframe(periods=365)

print(future.tail())

X_data = pd.DataFrame(data={'ds': ['2020-02-15', '2019-02-14']})


forecast = m.predict(X_data)

future_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(365)
print(future_forecast.shape)
print(future_forecast)

# Python
# fig1 = m.plot(forecast)


# Python
# fig2 = m.plot_components(forecast)

# yhat
