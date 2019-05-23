from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import traffic
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import traffic
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score

# load data

traffic_data = traffic.load_traffic(pipeline="full", scale=False)

parameters = {
        'max_depth': [6, 8],
        "learning_rate": [0.01],
        "n_estimators": [50, 80, 100, 200, 300],
        'min_child_weight': [1, 5],
        'gamma': [0, 0.2, 0.5, 1],
        'subsample': [0.6, 0.8, 1],
        'colsample_bytree': [0.8, 1],
        "scale_pos_weight": [1, 1.5, 2, 2.5, 3, 3.5, 4],
        "verbosity": [3],
        "random_state": [42]
        #'tree_method':['gpu_hist'],
        #'predictor':['gpu_predictor']
        }


SCORING = ['f1']
for x in SCORING:
    clf = GridSearchCV(XGBClassifier(), parameters, cv=5, n_jobs=-1, scoring=x, verbose=True)
    clf.fit(traffic_data.data, traffic_data.target)
    #print(clf.score(traffic_data.data, traffic_data.target))
    #print(str(clf.best_params_))

    with open("xgb/scores_boosting_xg_f1.txt", "a") as f:
        f.write(str(x) + ", SCORE: " + str(clf.score(traffic_data.data, traffic_data.target)) + ", PARAMS: " + str(clf.best_params_) + "\n")
            
    print("Done with ", x)

