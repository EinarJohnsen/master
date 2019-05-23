import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import numpy as npimport pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import cross_validate
from xgboost import plot_importance, to_graphviz, plot_tree
import traffic_spec
import operator 
from scipy import interp
from inspect import signature

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

f = plt.figure()


# Code based on: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py
def plot_precision_recall_curve(y_target, y_prob):
   average_precision = average_precision_score(y_target, y_prob)

   baseline = sum(y_target)/len(y_target)
   precision, recall, _ = precision_recall_curve(y_target, y_prob)

   step_kwargs = ({'step': 'post'}
                  if 'step' in signature(plt.fill_between).parameters
                  else {})
   plt.step(recall, precision, color='b', alpha=0.2, where='post')
   plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

   plt.hlines(baseline, xmin=0, xmax=1, colors='r', linestyles='dashed')

   plt.xlabel('Recall')
   plt.ylabel('Precision')
   plt.ylim([0.0, 1.05])
   plt.xlim([0.0, 1.0])
   plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

   plt.show()




spec_files = [None, 'data_full/data_k10_full.txt', 'data_full/data_k20_full.txt', 'data_full/data_k30_full.txt']
spec_train_file = None
spec_test_file = None
i = 0 
for spec_file in spec_files:
    if spec_file is not None:
        spec = np.genfromtxt(spec_file)
        train = spec[:42266, :]
        test = spec[42266:, :]
        np.savetxt('spec_train_1.txt', train)
        np.savetxt('spec_test_1.txt', test)
        spec_train_file = 'spec_train_1.txt'
        spec_test_file = 'spec_test_1.txt'
    train_data = traffic_spec.load_traffic(pipeline='full', scale=False, spectral_path=spec_train_file)
    test_data = traffic_spec.load_traffic(filepath='test_raw.csv', pipeline='full', scale=False, spectral_path=spec_test_file)


    X_train, y_train = train_data.data, train_data.target
    X_test, y_test = test_data.data, test_data.target

    param_dist =  {'colsample_bytree': 0.8, 'gamma': 1, 'learning_rate': 0.15, 'max_depth': 4, 'min_child_weight': 5, 'n_estimators': 50, 'scale_pos_weight': 4, 'subsample': 0.8, 'verbosity': 3, "random_state": 42}
    
    clf = XGBClassifier(**param_dist)

    clf.fit(X_train, y_train)

    ddd = clf.predict(X_test)
    eee = clf.predict_proba(X_test)

            # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, ddd)
    print('Accuracy: %f' % accuracy)
        # precision tp / (tp + fp)
    precision = precision_score(y_test, ddd)
    print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
    recall = recall_score(y_test, ddd)
    print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, ddd)
    print('F1 score: %f' % f1)


    fff = [e[1] for e in eee]
    
    roc_auc = roc_auc_score(y_test, fff)
    print('ROC AUC score: %f' % roc_auc)


    avg_prec = average_precision_score(y_test, fff)
    print('Avg Precision: %f' % avg_prec)
    
    balanced_acc = balanced_accuracy_score(y_test, ddd)
    print('Balanced Accuracy: %f' % balanced_acc)

    new_dict = {x:y*100 for x,y in zip(train_data.feature_names, clf.feature_importances_)}

    new_dict = sorted(new_dict.items(),key = operator.itemgetter(1),reverse = True)

    for x in new_dict[:15]:
            print(x)
    
    # Code to plot: based on https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    """
    fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc_ = auc(fpr, tpr)
    aucs.append(roc_auc_)
    plt.plot(fpr, tpr, lw=1, alpha=0.4,
             label='ROC k=%d (AUC = %0.2f)' % ((i)*10, roc_auc))

    i += 1
    
    #plot_precision_recall_curve(y_test, clf.predict_proba(X_test)[:, 1])
    """

"""
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.5)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.3,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
"""

