import matplotlib.pyplot as plt
from itertools import cycle
import seaborn as sns
from inspect import signature
from sklearn.metrics import precision_recall_curve, average_precision_score
import random
import numpy


def plot_roc_auc(fpr,tpr,roc_auc,n_classes):
    # TODO: get the output as PDF
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i + 1, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class classification')
    plt.legend(loc="lower right")
    plt.savefig("ROC_Curve.pdf", format="pdf")
    plt.show()


def plot_precision_recall(ground_truth_matrix, similarity_matrix):
    N = len(ground_truth_matrix)
    ground_truth_matrix = []
    for i in range(N):
        row = []
        for j in range(N):
            temp = random.randint(0, 1)
            row.append(temp)
        ground_truth_matrix.append(row)
    ground_truth_matrix = numpy.asarray(ground_truth_matrix, dtype=int)
    precision, recall, _ = precision_recall_curve(ground_truth_matrix.flatten(), similarity_matrix.flatten())
    average_precision = average_precision_score(ground_truth_matrix.flatten(), similarity_matrix.flatten())
    plt.figure()
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))
    plt.savefig("Precision_Recall_Curve.pdf", format="pdf")
    plt.show()


def plot_scatter_matrix(ground_truth_matrix, similarity_matrix ):
    plt.figure()
    sns.scatterplot(x=ground_truth_matrix.flatten(), y=similarity_matrix.flatten(), hue=ground_truth_matrix.flatten())
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.title('Scatter plot of true and predicted labels')
    plt.savefig("Scatter_Plot.pdf", format="pdf")
    plt.show()


def plot_ndcg(ndcg):
    plt.figure()
    k_values = [1]
    plt.plot(k_values, ndcg)
    plt.xlabel('Number of recommended items (k)')
    plt.ylabel('NDCG score')
    plt.title('NDCG curve')
    plt.savefig("NDCG_Curve.pdf", format="pdf")
    plt.show()


def plot_box_plot(matrix):
    # assume similarities is a pandas DataFrame containing similarity scores for each item
    plt.figure()
    boxplot = matrix.boxplot()
    plt.savefig('Boxplot.pdf', rot=45, format="pdf")
    plt.show()


def plot_heatmap(matrix):
    plt.figure()
    sns.heatmap(matrix, annot=True)
    plt.savefig('Heatmap.pdf', format="pdf")
    plt.show()


