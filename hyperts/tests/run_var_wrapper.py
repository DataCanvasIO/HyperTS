import pandas as pd

from hyperts.hyper_ts import VARWrapper
from random import random
import datetime

now_date = datetime.datetime.now()


# contrived dataset with dependency
data = list()
X_train = []
for i in range(100):
    now_date = now_date + datetime.timedelta(days=1)
    X_train.append(now_date)
    v1 = i + random()
    v2 = v1 + random()
    row = [v1, v2]
    data.append(row)

# fit model
X_train = pd.DataFrame(data={'ds': X_train})
print(X_train)

# y_train
y_train = pd.DataFrame(data=data)
y_train.columns = ['var_1', 'var_2']
print(y_train)


model = VARWrapper()
model.fit(X_train, y_train)

test_data = []
for i in range(60):
    now_date = now_date + datetime.timedelta(days=1)
    test_data.append(now_date)

X_test = pd.DataFrame(data={'ds': test_data})


y_pred = model.predict(X_test)
print(y_pred)


