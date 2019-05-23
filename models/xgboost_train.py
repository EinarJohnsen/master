from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate
from matplotlib import pyplot
from xgboost import plot_importance, to_graphviz, plot_tree
import traffic_spec
import operator 


traffic_data = traffic_spec.load_traffic(pipeline="full", scale=False, spectral_path=None)

X_train, X_test, y_train, y_test = train_test_split(traffic_data.data, traffic_data.target, test_size=0.1, random_state=42)


param_dist =  {'colsample_bytree': 0.8, 'gamma': 1, 'learning_rate': 0.15, 'max_depth': 4, 'min_child_weight': 5, 'n_estimators': 50, 'scale_pos_weight': 4, 'subsample': 0.8, 'verbosity': 3, "random_state": 42}

clf = XGBClassifier(**param_dist)

cross_v = cross_validate(clf, traffic_data.data, traffic_data.target, scoring=['accuracy', 'balanced_accuracy', 'average_precision', 'roc_auc', 'recall', 'precision', 'f1'], cv=5)

for y in cross_v:
    if "test" in y:
        print(y, " : ", np.mean(cross_v[y]))
