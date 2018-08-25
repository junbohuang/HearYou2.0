import os
import itertools

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout(h_pad=1.08)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_cm(model_name, emotions_used, Y, prediction, data_type, feature_type):

    y_pred = prediction.argmax(1)
    y_true = Y.argmax(1)

    # confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    np.set_printoptions(precision=2)
    print(cnf_matrix)

    # Plot non-normalized confusion matrix
    path_to_cm = './plots/' + model_name

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=emotions_used,
                          title='Confusion Matrix')
    cm_name = "/cm_" + model_name + "_" + data_type + "_" + feature_type + '.png'
    plt.savefig(path_to_cm + cm_name)

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=emotions_used, normalize=True,
                          title='Normalized Confusion Matrix')
    cm_name = "/Normalized_cm_" + model_name + "_" + data_type + "_" + feature_type + '.png'
    plt.savefig(path_to_cm + cm_name)
    plt.close()