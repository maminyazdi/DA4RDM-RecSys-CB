import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, mean_squared_error, roc_curve, ndcg_score, classification_report ,auc , roc_auc_score
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from src.distance_similarity_calculator import result_function
# from warnings import filterwarnings
from src.roc import compute_ROC

def evaluate(df):
    data = pd.read_csv("../Data/tomography.csv", sep='|')
    resource_list = data["Resource"].unique().tolist()

    resource_data_dict = {}
    for resource in resource_list:
        matrix = result_function(df, resource, distanceMethod='euclidean', outputFormatJson=False, DEBUG_MODE=False)
        conditions = [
            (matrix['distance'] < 0.2),
            (matrix['distance'] >= 0.2) & (matrix['distance'] < 0.4),
            (matrix['distance'] >= 0.4) & (matrix['distance'] < 0.6),
            (matrix['distance'] >= 0.6) & (matrix['distance'] < 0.8),
            (matrix['distance'] >= 0.8)
        ]
        values = [5, 4, 3, 2, 1]
        matrix['distance'] = np.select(conditions, values)
        resource_data_dict.update({resource: matrix})
    resource_list.append("Resource")

    eval_matrix = pd.DataFrame(columns=[resource_list])

    for resource, data in resource_data_dict.items():
        eval_matrix.loc[-1, ["Resource"]] = resource
        for index, rows in data.iterrows():
            eval_matrix.loc[-1, [index]] = rows["distance"]
        eval_matrix.index = eval_matrix.index + 1

    eval_matrix.set_index('Resource', inplace=True)
    


    #Converting resource names into an array of int
    ground_truth_matrix = eval_matrix.to_numpy(dtype=int)
    similarity_matrix = eval_matrix.to_numpy(dtype=int)

    # TODO: Export the ground_truth_matrix as CSV (maybe before converting to array)

    confusion = confusion_matrix(ground_truth_matrix.flatten(), similarity_matrix.flatten())
    # Compute ROC
    compute_ROC(ground_truth_matrix, similarity_matrix,confusion)

    # Compute precision, recall, and F1 score
    precision, recall, f1_score, _ = precision_recall_fscore_support(ground_truth_matrix.flatten(),
                                                                     similarity_matrix.flatten(), average='macro')

    # Compute mean squared error
    mse = mean_squared_error(ground_truth_matrix.flatten(), similarity_matrix.flatten())

    # Compute NDCG score
    ndcg = ndcg_score(ground_truth_matrix, similarity_matrix)
    
    # Compute Spearman's Rank Correlation Coefficient
    spearman_corr, _ = spearmanr(ground_truth_matrix.flatten(), similarity_matrix.flatten())
    
    print("Confusion matrix:\n", confusion)
    print("classification_report: ", classification_report(ground_truth_matrix.flatten(), similarity_matrix.flatten()))
    print("Precision: {:.4f}, Recall: {:.4f}, F1 score: {:.4f}".format(precision, recall, f1_score))
    print("Mean squared error: {:.4f}".format(mse))
    print("NDCG score: {:.4f}".format(ndcg))
    print("Spearman's Rank Correlation Coefficient: {:.4f}".format(spearman_corr))
