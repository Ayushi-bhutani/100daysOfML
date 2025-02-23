import numpy as np
from sklearn import linear_model
n_samples, n_features = 1, 500
y = np.random.randn(n_samples)
x = np.random.randn(n_samples, n_features)
clf = linear_model.SGDRegressor()
import time 
start_time = time.time()
clf.partial_fit(x,y)
elapsed_time = time.time() - start_time
print(elapsed_time)
clf.partial_fit(x,y)
print(elapsed_time)
