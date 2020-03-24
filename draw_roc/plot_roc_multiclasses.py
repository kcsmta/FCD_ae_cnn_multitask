import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
import sys

import common

show_roc= False
save_roc = True
fig_extension ='pdf'

colors = cycle(['deeppink', 'lawngreen','b','darkorange', 'purple'])


def plot_roc(roc_file_path, save_to):
    resultfile = roc_file_path
    print('read data from:', resultfile)
    curve_labels, y_test, y_score = common.readROCFile(results=resultfile, keepFormat=True)
    # list of classes
    classes = np.unique(y_test)
    print('classes: ', classes)
    y_test = label_binarize(y_test, classes=classes)
    n_classes = y_test.shape[1]
    print('n_class', n_classes)
    y_score = np.array(y_score)
    print(y_test[0:2])
    print(y_score[0:2])
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        y = y_test[:, i]
        score = y_score[:, i]
        fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_score[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    lw = 2
    ##############################################################################
    # Plot ROC curves for the multiclass problem

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='r', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('')
    plt.legend(loc="lower right")
    if save_roc:
        basename = os.path.basename(resultfile).split(".")[0] + "." + fig_extension
        plt.savefig(os.path.join(save_to, basename))
        print('ROC is save to:', os.path.join(save_to, basename))
    if show_roc:
        plt.show()

import os
folder_path = "../ProgramClassification/results"
save_to = "./roc_result"
file_path_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.find(".roc")>-1]
for file in file_path_list:
    plot_roc(file, save_to)
