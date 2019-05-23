
""" Random forest - Gridsearch """

from multiprocessing import cpu_count

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate

import dt_rf
import traffic

NUM_FOLDS = 5
N_JOBS = cpu_count() - 1
print(f'Using {N_JOBS} cores..')

params = dt_rf.params

data = traffic.load_traffic(pipeline='full', scale=False)

param_grid = {'n_esimators': np.arange(1, 1000)}

gs = GridSearchCV(RandomForestClassifier(**params, random_state=42),
                  param_grid, cv=5, scoring='average_precision', n_jobs=N_JOBS, verbose=1)
gs.fit(data.data, data.target)

print(gs.best_params_)
print(gs.best_estimator_)
