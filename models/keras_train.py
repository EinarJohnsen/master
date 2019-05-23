import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import traffic
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pickle
import data_getter
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
import traffic2
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.constraints import unit_norm, min_max_norm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score, roc_curve, auc, balanced_accuracy_score, average_precision_score
import traffic_spec
import data_getter
from scipy import interp

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from inspect import signature


traffic_data = traffic_spec.load_traffic(pipeline="full", scale=False, scaler=StandardScaler, spectral_path=None)

Y = traffic_data.target

sc = StandardScaler()
X = sc.fit_transform(traffic_data.data)

print(traffic_data.data.shape)


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

prec_l = []
rec_l = []
f1_l = []
cvscores_l = []
roc_auc_l = []
average_precision_l = []
balanced_accuracy_l = []


tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

f = plt.figure()

i = 0
for train, test in kfold.split(X, Y):

    classifier = Sequential()
    classifier.add(Dense(50, activation='relu', input_dim=len(X[train][0])))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(50, activation='relu'))
    classifier.add(Dense(1, activation='sgmoid'))

    class_weight = {0 : 1., 1: 4}

    classifier.compile(optimizer ='adam', loss='binary_crossentropy', metrics =['accuracy'])

    history = classifier.fit(X[train], Y[train], batch_size=30, epochs=15, class_weight=class_weight, validation_data=(X[test], Y[test]))


    # predict probabilities for test set
    yhat_probs = classifier.predict(X[test], verbose=0)
    print(yhat_probs)
    # predict crisp classes for test set
    yhat_classes = classifier.predict_classes(X[test], verbose=0)
    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]

    accuracy = accuracy_score(Y[test], yhat_classes)
    print('Accuracy: %f' % accuracy)
    precision = precision_score(Y[test], yhat_classes)
    print('Precision: %f' % precision)
    recall = recall_score(Y[test], yhat_classes)
    print('Recall: %f' % recall)
    f1 = f1_score(Y[test], yhat_classes)
    print('F1 score: %f' % f1)
    roc_auc = roc_auc_score(Y[test], yhat_probs)
    print('ROC AUC score: %f' % roc_auc)


    balanced_accuracy = balanced_accuracy_score(Y[test], yhat_classes)
    average_precision = average_precision_score(Y[test], yhat_probs)


    cvscores_l.append(accuracy * 100)
    f1_l.append(f1 * 100)
    prec_l.append(precision * 100)
    rec_l.append(recall * 100)
    roc_auc_l.append(roc_auc * 100)
    average_precision_l.append(average_precision * 100)
    balanced_accuracy_l.append(balanced_accuracy * 100)

    print(sum(yhat_classes), sum(Y[test]))



print(" ")
print(" ")
print("acc: ","%.2f%% (+/- %.2f%%)" % (np.mean(cvscores_l), np.std(cvscores_l)))
print("*"*10)
print("f1: ","%.2f%% (+/- %.2f%%)" % (np.mean(f1_l), np.std(f1_l)))
print("*"*10)
print("precision: ","%.2f%% (+/- %.2f%%)" % (np.mean(prec_l), np.std(prec_l)))
print("*"*10)
print("recall: ","%.2f%% (+/- %.2f%%)" % (np.mean(rec_l), np.std(rec_l)))
print("*"*10)
print("ROC: ","%.2f%% (+/- %.2f%%)" % (np.mean(roc_auc_l), np.std(roc_auc_l)))
print("*"*10)
print("Balanced Acc: ","%.2f%% (+/- %.2f%%)" % (np.mean(balanced_accuracy_l), np.std(balanced_accuracy_l)))
print("*"*10)
print("Average Prec: ","%.2f%% (+/- %.2f%%)" % (np.mean(average_precision_l), np.std(average_precision_l)))

