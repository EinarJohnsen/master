
"""  Precision-recall curve """

import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.utils.fixes import signature


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
    plt.title(
        '2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

    plt.show()
