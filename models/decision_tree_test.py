
import os
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (cross_val_predict, cross_val_score,
                                     cross_validate, train_test_split)
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from plot_confusion_matrix import plot_confusion_matrix
from precision_recall_curve import plot_precision_recall_curve
from traffic import load_traffic

NUM_FOLDS = 5

metrics = [accuracy_score, balanced_accuracy_score, average_precision_score,
           roc_auc_score, recall_score, precision_score, f1_score]

# Optimal f1
params = {'class_weight': {0: 0.30000000000000004, 1: 1}, 'criterion': 'entropy',
          'max_depth': 6, 'max_features': None, 'min_samples_leaf': 9, 'min_samples_split': 2}


def main(spec_file):
    print(spec_file)
    #clf = DecisionTreeClassifier(**params, random_state=42)
    clf = RandomForestClassifier(
        **params, n_estimators=180, random_state=42, n_jobs=-1)
    #scores = cross_validate(clf, X, y, scoring=scoring, cv=NUM_FOLDS, return_train_score=False)

    res = {metric: [] for metric in metrics}
    for _ in range(1):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_score = clf.predict_proba(X_test)
        for metric in metrics:
            if metric == average_precision_score or metric == roc_auc_score:
                res[metric].append(metric(y_test, y_score[:, 1]))
            else:
                res[metric].append(metric(y_test, y_pred))

    for metric, results in res.items():
        print(f'{metric.__name__:25} {sum(results) / len(results) * 100:.1f}%')


if __name__ == '__main__':
    spec_files = [None, 'data_k10_full.txt',
                  'data_k20_full.txt', 'data_k30_full.txt']
    spec_train_file = None
    spec_test_file = None
    for spec_file in spec_files:
        if spec_file is not None:
            spec = np.genfromtxt(spec_file)
            train = spec[:42266, :]
            test = spec[42266:, :]
            np.savetxt('spec_train.txt', train)
            np.savetxt('spec_test.txt', test)
            spec_train_file = 'spec_train.txt'
            spec_test_file = 'spec_test.txt'
        train_data = load_traffic(
            pipeline='full', scale=False, spectral_path=spec_train_file)
        test_data = load_traffic(
            filepath='test_raw.csv', pipeline='full', scale=False, spectral_path=spec_test_file)
        X_train, y_train = train_data.data, train_data.target
        X_test, y_test = test_data.data, test_data.target
        main(spec_file)
