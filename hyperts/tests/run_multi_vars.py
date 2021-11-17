from statsmodels.tsa.api import VAR
# VAR example
from statsmodels.tsa.vector_ar.var_model import VAR
from random import random
# VAR example
from statsmodels.tsa.vector_ar.var_model import VAR
from random import random
# contrived dataset with dependency
data = list()
for i in range(100):
    v1 = i + random()
    v2 = v1 + random()
    row = [v1, v2]
    data.append(row)

# fit model
model = VAR(data)
model_fit = model.fit()

# make prediction
yhat = model_fit.forecast(model_fit.y, steps=10)
print(yhat)
