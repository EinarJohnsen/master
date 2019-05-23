
""" Decision tree - GridSearch"""

import time
from multiprocessing import cpu_count

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from traffic import load_traffic

# Use all cores but one
N_JOBS = cpu_count() - 1
print('> Using {} cores'.format(N_JOBS))

data = load_traffic(pipeline='full', scale=False)

PARAM_GRID = {'max_depth': np.arange(1, 10), 'criterion': ['gini', 'entropy'],
              'min_samples_split': np.arange(2, 10),
              'min_samples_leaf': np.arange(1, 10),
              'class_weight': [{0: n, 1: 1} for n in np.arange(0.1, 1.01, 0.1)],
              'max_features': ['auto', 'sqrt', 'log2', None]}

# Add balanced weight to class_weight
PARAM_GRID['class_weight'].append(
    {0: 0.2738862696045415, 1: 0.7261137303954585})

SCORING = ['accuracy', 'balanced_accuracy',
           'average_precision', 'roc_auc', 'recall', 'precision', 'f1']


def get_params(scoring):
    gs = GridSearchCV(DecisionTreeClassifier(random_state=42), PARAM_GRID, cv=5,
                      scoring=scoring, n_jobs=N_JOBS)
    gs.fit(data.data, data.target)
    with open('decision_tree_params.txt', 'a') as f:
        f.write(str(scoring) + ':' + str(gs.best_params_) + '\n')


for score in SCORING:
    get_params(score)
