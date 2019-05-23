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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.constraints import unit_norm, min_max_norm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import average_precision_score
import traffic_spec

from sklearn.metrics import f1_score, roc_curve, auc, balanced_accuracy_score, average_precision_score
from scipy import interp

spec_files = [None, 'data_full/data_k10_full.txt', 'data_full/data_k20_full.txt', 'data_full/data_k30_full.txt']
spec_train_file = None
spec_test_file = None


accuracy_1 = []
precision_1 = []
recall_1 = []
average_precision_1 = []
balanced_accuracy_1 = []
f1_1 = []
roc_auc_1 = []

accuracy_2 = []
precision_2 = []
recall_2 = []
average_precision_2 = []
balanced_accuracy_2 = []
f1_2 = []
roc_auc_2 = []

accuracy_3 = []
precision_3 = []
recall_3 = []
average_precision_3 = []
balanced_accuracy_3 = []
f1_3 = []
roc_auc_3 = []

accuracy_4 = []
precision_4 = []
recall_4 = []
average_precision_4 = []
balanced_accuracy_4 = []
f1_4 = []
roc_auc_4 = []

i = 0
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

f = plt.figure()


for x in range(0, 5):
    counter = 1
    for spec_file in spec_files:
        if spec_file is not None:
            spec = np.genfromtxt(spec_file)
            train = spec[:42266, :]
            test = spec[42266:, :]
            np.savetxt('spec_train.txt', train)
            np.savetxt('spec_test.txt', test)
            spec_train_file = 'spec_train.txt'
            spec_test_file = 'spec_test.txt'
        train_data = traffic_spec.load_traffic(pipeline='full', scale=False, spectral_path=spec_train_file)
        test_data = traffic_spec.load_traffic(filepath='test_raw.csv', pipeline='full', scale=False, spectral_path=spec_test_file)


        X_train, y_train = train_data.data, train_data.target
        X_test, y_test = test_data.data, test_data.target


        sc = StandardScaler()
        X = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)


        classifier = Sequential()
        classifier.add(Dense(50, activation='relu', input_dim=len(X[0])))
        classifier.add(Dropout(0.1))
            #Second  Hidden Layer
        classifier.add(Dense(50, activation='relu'))

        classifier.add(Dense(1, activation='sigmoid'))

        

        #Compiling the neural network
        classifier.compile(optimizer ='adam', loss='binary_crossentropy', metrics =['accuracy'])

        #class_weight = {0 : 1., 1: 4.5}
        class_weight = {0 : 1., 1: 4}

        history = classifier.fit(X, y_train, batch_size=30, epochs=15, class_weight=class_weight, validation_data=(X_test, y_test))

        
            # predict probabilities for test set
        yhat_probs = classifier.predict(X_test, verbose=0)
            # predict crisp classes for test set
        yhat_classes = classifier.predict_classes(X_test, verbose=0)
            # reduce to 1d array
        yhat_probs = yhat_probs[:, 0]
        yhat_classes = yhat_classes[:, 0]

        accuracy = accuracy_score(y_test, yhat_classes)
        precision = precision_score(y_test, yhat_classes)
        recall = recall_score(y_test, yhat_classes)
        f1 = f1_score(y_test, yhat_classes)
        roc_auc = roc_auc_score(y_test, yhat_probs)
        balanced_accuracy = balanced_accuracy_score(y_test, yhat_classes)
        average_precision = average_precision_score(y_test, yhat_probs)


        if counter == 1:
            accuracy_1.append(accuracy)
            precision_1.append(precision)
            recall_1.append(recall)
            f1_1.append(f1)
            roc_auc_1.append(roc_auc)
            balanced_accuracy_1.append(balanced_accuracy)
            average_precision_1.append(average_precision)

        if counter == 2:
            accuracy_2.append(accuracy)
            precision_2.append(precision)
            recall_2.append(recall)
            f1_2.append(f1)
            roc_auc_2.append(roc_auc)
            balanced_accuracy_2.append(balanced_accuracy)
            average_precision_2.append(average_precision)

        if counter == 3:
            accuracy_3.append(accuracy)
            precision_3.append(precision)
            recall_3.append(recall)
            f1_3.append(f1)
            roc_auc_3.append(roc_auc)
            balanced_accuracy_3.append(balanced_accuracy)
            average_precision_3.append(average_precision)

        if counter == 4:
            accuracy_4.append(accuracy)
            precision_4.append(precision)
            recall_4.append(recall)
            f1_4.append(f1)
            roc_auc_4.append(roc_auc)
            balanced_accuracy_4.append(balanced_accuracy)
            average_precision_4.append(average_precision)

        counter += 1 




print("*"*50)
print(np.mean(accuracy_1), " acc")
print(np.mean(precision_1), " precision")
print(np.mean(recall_1), " recall")
print(np.mean(f1_1), " f1")
print(np.mean(roc_auc_1), " roc_auc")
print(np.mean(balanced_accuracy_1), " balanced_acc")
print(np.mean(average_precision_1), " average_precision")
print("*"*50)
print(np.mean(accuracy_2), " acc")
print(np.mean(precision_2), " precision")
print(np.mean(recall_2), " recall")
print(np.mean(f1_2), " f1")
print(np.mean(roc_auc_2), " roc_auc")
print(np.mean(balanced_accuracy_2), " balanced_acc")
print(np.mean(average_precision_2), " average_precision")
print("*"*50)
print(np.mean(accuracy_3), " acc")
print(np.mean(precision_3), " precision")
print(np.mean(recall_3), " recall")
print(np.mean(f1_3), " f1")
print(np.mean(roc_auc_3), " roc_auc")
print(np.mean(balanced_accuracy_3), " balanced_acc")
print(np.mean(average_precision_3), " average_precision")
print("*"*50)
print(np.mean(accuracy_4), " acc")
print(np.mean(precision_4), " precision")
print(np.mean(recall_4), " recall")
print(np.mean(f1_4), " f1")
print(np.mean(roc_auc_4), " roc_auc")
print(np.mean(balanced_accuracy_4), " balanced_acc")
print(np.mean(average_precision_4), " average_precision")


