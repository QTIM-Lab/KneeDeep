import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
from .io.image import save_figure


def dice(y_true, y_pred):

    y_true_f = y_true[:]
    y_pred_f = y_pred[:]
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f))


def roc_analysis(y_true, y_pred, save_path=None, label='test'):
    """
    ROC analysis for segmentation
    :param y_true:
    :param y_pred:
    :param save_path:
    :param label:
    :return:
    """

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    youden = optimal_threshold(tpr, 1 - fpr)
    auroc = auc(fpr, tpr)

    lw = 2
    plt.plot(fpr, tpr, lw=lw, label='{} (AUC = {:.2f})'.format(label.capitalize(), auroc))

    if save_path is not None:
        plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        save_figure(save_path)

    return tpr, 1 - fpr, auroc, youden, thresholds[youden]


def optimal_threshold(sensitivity, specificity):

    j = sensitivity + specificity - 1
    return np.argmax(j)
