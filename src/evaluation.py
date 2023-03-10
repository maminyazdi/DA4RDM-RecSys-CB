import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, mean_squared_error, roc_curve, \
    ndcg_score, classification_report, auc, roc_auc_score , mean_absolute_error, silhouette_samples, silhouette_score
from scipy.stats import spearmanr
import numpy as np
from src.distance_similarity_calculator import result_function
# from warnings import filterwarnings
from src.roc import compute_ROC
from src.plotting import plot_ndcg , plot_box_plot, plot_heatmap, plot_silhouette_score, plot_scatter_matrix, plot_precision_recall


def evaluate(df):
    # DO NOT DELETE THE BELLOW LINE OF CODE
    # eval_matrix, ground_truth_matrix, similarity_matrix = create_resources_similarity_matrix(df)


    for i in range(2):

        ground_truth_matrix, similarity_matrix = get_matrixes(i)

        if i == 0:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>> Cosine <<<<<<<<<<<<<<<<<<<<<<<<<<<")
        else:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>> Euclidean <<<<<<<<<<<<<<<<<<<<<<<<")
        confusion = confusion_matrix(ground_truth_matrix.flatten(), similarity_matrix.flatten())

        # calculate silhouette scores for each sample in the confusion matrix
        silhouette_vals = silhouette_samples(similarity_matrix, np.argmax(similarity_matrix, axis=1))
        # calculate mean silhouette score for the entire dataset
        silhouette_avg = silhouette_score(similarity_matrix, np.argmax(similarity_matrix, axis=1))

        plot_silhouette_score(silhouette_vals,silhouette_avg,similarity_matrix)

        # Compute ROC
        compute_ROC(ground_truth_matrix, similarity_matrix, confusion)
        # Compute precision, recall, and F1 score
        precision, recall, f1_score, _ = precision_recall_fscore_support(ground_truth_matrix.flatten(),
                                                                         similarity_matrix.flatten(), average='macro')
        # Compute mean squared error (MSE)
        mse = mean_squared_error(ground_truth_matrix.flatten(), similarity_matrix.flatten())
        rmse = mean_squared_error(ground_truth_matrix.flatten(), similarity_matrix.flatten(), squared=False)
        # compute mean absolute error (MAE)
        mae = mean_absolute_error(ground_truth_matrix.flatten(),similarity_matrix.flatten())
        # Compute NDCG score
        ndcg = ndcg_score(ground_truth_matrix, similarity_matrix)
        plot_ndcg(ndcg)

        # Compute Spearman's Rank Correlation Coefficient
        spearman_corr, _ = spearmanr(ground_truth_matrix.flatten(), similarity_matrix.flatten())
        # plot_scatter_matrix(ground_truth_matrix, similarity_matrix)
        # plot_precision_recall(ground_truth_matrix, similarity_matrix)


        eval_matrix = pd.DataFrame(similarity_matrix)
        #eval_matrix = eval_matrix.astype(int)
        plot_box_plot(eval_matrix)
        plot_heatmap(eval_matrix)

        # print("Confusion matrix:\n", confusion)
        print("classification_report: \n", classification_report(ground_truth_matrix.flatten(), similarity_matrix.flatten()))
        #print("Precision: {:.4f}, Recall: {:.4f}, F1 score: {:.4f}".format(precision, recall, f1_score))
        print("Mean Squared Error (MSE): {:.4f}".format(mse))
        print("Root Mean Squared Error (RMSE): {:.4f}".format(rmse))
        print("Mean Average Error (MAE): {:.4f}".format(mae))
        print("NDCG score: {:.4f}".format(ndcg))
        print("Spearman's Rank Correlation Coefficient: {:.4f}".format(spearman_corr))
        print("silhouette_vals score:\n", silhouette_vals)
        print("silhouette_avg score: ",silhouette_avg)


def get_matrixes(index):
    if index == 0:
        #cosine
        similarity_matrix = pd.read_csv(
            "C:/Users/ay-admin/PycharmProjects/da4rdm-recsys-cb/Data/Evaluation/Cosine/similarity_matrix_cosine.csv")
        similarity_matrix.drop(columns=["Unnamed: 0"], inplace=True)
        similarity_matrix = similarity_matrix.to_numpy(dtype=int)
    else:
        #Euclidean
        similarity_matrix = pd.read_csv(
            "C:/Users/ay-admin/PycharmProjects/da4rdm-recsys-cb/Data/Evaluation/Euclidean/similarity_matrix_euclidean.csv")
        similarity_matrix.drop(columns=["Unnamed: 0"], inplace=True)
        similarity_matrix = similarity_matrix.to_numpy(dtype=int)

    ground_truth_matrix = pd.read_csv(
        "C:/Users/ay-admin/PycharmProjects/da4rdm-recsys-cb/Data/Evaluation/ground_truth/ground_truth_matrix.csv")
    ground_truth_matrix.drop(columns=["Unnamed: 0"], inplace=True)
    ground_truth_matrix = ground_truth_matrix.to_numpy(dtype=int)

    return ground_truth_matrix, similarity_matrix



def make_symmetric(asym_matrix):
    # Get the dimensions of the asymmetric matrix
    nrows, ncols = asym_matrix.shape

    # Create an empty symmetric matrix
    sym_matrix = np.zeros((nrows, ncols), dtype=int)

    # Copy the lower triangle to the upper triangle
    for i in range(nrows):
        for j in range(i):
            sym_matrix[i][j] = asym_matrix[j][i]
            sym_matrix[j][i] = asym_matrix[j][i]

        # Copy the diagonal element
        sym_matrix[i][i] = asym_matrix[i][i]

    return sym_matrix


def create_resources_similarity_matrix(df):
    data = pd.read_csv("C:/Users/ay-admin/PycharmProjects/da4rdm-recsys-cb/Data/tomography-new.csv", sep='|')
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
    eval_matrix.to_csv(
        "C:/Users/ay-admin/PycharmProjects/da4rdm-recsys-cb/Data/Evaluation/Euclidean/eval_matrix_euclidean.csv")
    # Converting resource names into an array of int
    ground_truth_matrix = eval_matrix.to_numpy(dtype=int)
    ground_truth_matrix = make_symmetric(ground_truth_matrix)


    similarity_matrix = eval_matrix.to_numpy(dtype=int)
    similarity_matrix = make_symmetric(similarity_matrix)
    pd.DataFrame(similarity_matrix).to_csv(
        "C:/Users/ay-admin/PycharmProjects/da4rdm-recsys-cb/Data/Evaluation/Euclidean/similarity_matrix_euclidean.csv")




    return eval_matrix, ground_truth_matrix, similarity_matrix


