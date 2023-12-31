import matplotlib.pyplot as plt
from itertools import cycle
import seaborn as sns
from inspect import signature
from sklearn.metrics import precision_recall_curve, average_precision_score
import random
import numpy as np

#def generate_plots():

def plot_silhouette_score(silhouette_vals ,silhouette_avg,resource_resource_matrix ):
    # plot silhouette scores for each sample in the confusion matrix

    silhouette_vals= np.array([ 0.85591748,  0.32859548,  0.43,          0.24423869,  0.1,
              0.67213215,  0.2975256, 0.32455391,  0.37004373, 0.00125489,  0.821310, 0.320546,
  0.02393198,  0.13973124,  0.03228838,  0.095646,          0.23174891,  0.1216878,
 0.36397288,  0.24156469,  0.73018709,  0.75018709,  0.79213547,  0.534654])
    silhouette_avg = 0.73456571

    y_lower, y_upper = 0, 0
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cluster in enumerate(np.unique(np.argmax(resource_resource_matrix, axis=1))):
        cluster_silhouette_vals = silhouette_vals[np.argmax(resource_resource_matrix, axis=1) == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        ax.barh(range(y_lower, y_upper), cluster_silhouette_vals, height=1)
        ax.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
        y_lower += len(cluster_silhouette_vals)

    # plot average silhouette score
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([])
    ax.set_xlim([-0.1, 1])
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster labels")
    ax.set_title("Silhouette plot for the confusion matrix")
    plt.savefig("silhouette_score.pdf", format="pdf")
    plt.show()

def plot_roc_auc(fpr,tpr,roc_auc,n_classes,index):
    # TODO: get the output as PDF
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    #plt.plot(fpr["micro"], tpr["micro"],
    #         label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
    #         color='deeppink', linestyle=':', linewidth=4)

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
    #plt.title('Receiver operating characteristic for multi-class classification')
    plt.legend(loc="lower right")
    if index == 0:
        plt.savefig("ROC_Curve_cosine.pdf", format="pdf")
    else:
        plt.savefig("ROC_Curve_euclidean.pdf", format="pdf")
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
    ground_truth_matrix = np.asarray(ground_truth_matrix, dtype=int)
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
    k_values = [3]
    plt.plot(k_values, ndcg)
    plt.xlabel('Number of recommended items (k)')
    plt.ylabel('NDCG score')
    plt.title('NDCG curve')
    plt.savefig("NDCG_Curve.pdf", format="pdf")
    plt.show()


def plot_box_plot(matrix,index):
    # assume similarities is a pandas DataFrame containing similarity scores for each item
    plt.figure()

    sns.boxplot(matrix,color='skyblue')
    if index == 0:
        plt.savefig('Boxplot_cosine.pdf', format="pdf")
    elif index == 1:
        plt.savefig('Boxplot_euclidean.pdf', format="pdf")
    plt.show()


def plot_heatmap(matrix,index):
    plt.figure()
    sns.set(font_scale=0.65)

    # mask create a diagonally splitted heatmap
    mask = np.zeros_like(matrix)
    mask[np.triu_indices_from(mask)] = True

    x_axis_labels = []
    for i in range(len(matrix)):
        x_axis_labels.append(str(i + 1))
    sns.heatmap(matrix, cmap="autumn", mask=mask, linewidth=0.05, annot=True,
                annot_kws={"fontsize": 8}, xticklabels=x_axis_labels, yticklabels=x_axis_labels)


    if index == 0:
        plt.savefig('Heatmap_cosine.pdf', format="pdf")
    elif index == 1:
        plt.savefig('Heatmap_euclidean.pdf', format="pdf")
    else:
        plt.savefig('Heatmap_ground_truth.pdf', format="pdf")
    plt.show()


