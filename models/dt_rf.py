
from collections import Counter

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import (cross_val_predict, cross_val_score,
                                     cross_validate, train_test_split)
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from plot_confusion_matrix import plot_confusion_matrix
from precision_recall_curve import plot_precision_recall_curve
from traffic import load_traffic

NUM_FOLDS = 5

scoring = ['accuracy', 'balanced_accuracy',
           'average_precision', 'roc_auc', 'recall', 'precision', 'f1']

# Optimal average precision
#params = {'class_weight': {0: 0.2, 1: 1}, 'criterion': 'entropy', 'max_depth': 7, 'max_features': None, 'min_samples_leaf': 6, 'min_samples_split': 2}

# Optimal balanced accuracy
#params = {'class_weight': {0: 0.4, 1: 1}, 'criterion': 'gini', 'max_depth': 6, 'max_features': None, 'min_samples_leaf': 9, 'min_samples_split': 2}

# Optimal f1
params = {'class_weight': {0: 0.30000000000000004, 1: 1}, 'criterion': 'entropy',
          'max_depth': 6, 'max_features': None, 'min_samples_leaf': 9, 'min_samples_split': 2}

#params = {'class_weight': {0: 0.273, 1: 0.726}, 'criterion': 'entropy', 'max_depth': 6, 'max_features': None, 'min_samples_leaf': 9, 'min_samples_split': 2}

# Optimal recall
#params = {'class_weight': {0: 0.1, 1: 1}, 'criterion': 'gini', 'max_depth': 1, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2}

# Optimal precision
#params = {'class_weight': {0: 1.0, 1: 1}, 'criterion': 'gini', 'max_depth': 6, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 2}

# Optimal acc
#params = {'class_weight': {0: 0.9, 1: 1}, 'criterion': 'entropy', 'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 2}


def main(spec_file):
    print(spec_file)
    #clf = DecisionTreeClassifier(**params, random_state=42)
    clf = RandomForestClassifier(
        **params, n_estimators=180, random_state=42, n_jobs=-1)
    scores = cross_validate(clf, X, y, scoring=scoring,
                            cv=NUM_FOLDS, return_train_score=False)

    for score_name, score in scores.items():
        print(
            f'{score_name:25} , {score.mean() * 100:.1f}%') if score_name.startswith('test') else None

    # Run cv for plotting
    y_probs = cross_val_predict(
        clf, X, y, method='predict_proba', cv=NUM_FOLDS)
    #plot_precision_recall_curve(y, y_probs[:,1])

    # Confusion matrix
    y_pred = cross_val_predict(clf, X, y, cv=NUM_FOLDS)
    #plot_confusion_matrix(y, y_pred)

    # check correct pred by sample type
    sample_type_count = Counter(data.sample_type)
    pred_true_or_false = [len(set(y_ypred)) == 1 for y_ypred in zip(y, y_pred)]
    correct_by_sample_type = [sample_type for sample_type, correct in zip(
        data.sample_type, pred_true_or_false) if correct is True]
    correct_by_sample_type_count = Counter(correct_by_sample_type)

    print('correct ratio by sample_type')
    for sample_type in sample_type_count.keys():
        print(
            f'{sample_type:25} , {(correct_by_sample_type_count[sample_type] / sample_type_count[sample_type] * 100):.1f}%')

    correct_pred = [len(set(x)) == 1 for x in zip(y, y_pred)]
    correct = [e[0]
               for e in zip(data.sample_type, correct_pred) if e[1] is True]
    c = Counter(correct)

    print(c)
    print(Counter(data.sample_type))

    prod_clf = RandomForestClassifier(
        **params, n_estimators=180, random_state=42, n_jobs=-1)
    prod_clf.fit(X, y)
    for percent, param in sorted(zip(prod_clf.feature_importances_, data.feature_names), key=lambda e: e[0], reverse=True):
        print(f'{param} ({percent * 100:.1f}%)')
    print(len(prod_clf.feature_importances_))
    print('sum importance', sum(prod_clf.feature_importances_))
    print('='*20)


if __name__ == '__main__':
    spec_files = [None, 'data_k10.txt', 'data_k20.txt', 'data_k30.txt']
    for spec_file in spec_files:
        data = load_traffic(pipeline='full', scale=False,
                            spectral_path=spec_file)
        X, y = data.data, data.target
        main(spec_file)
