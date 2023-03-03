import pandas as pd

# from src.preprocessor import loadAndPreprocess_function
# from src.distance_similarity_calculator import result_function
from src.evaluation import evaluate

# df = loadAndPreprocess_function(filepath="C:/Users/ay-admin/PycharmProjects/da4rdm-recsys-cb/Data/tomography.csv", features=['http://purl.org/coscine/terms/sfb1394#annularMillingParameters'],debug=False)
# df.to_csv('DataProcessed_Tomography.csv')
# df = loadAndPreprocess_function(filepath="C:/Users/ay-admin/PycharmProjects/da4rdm-recsys-cb/Data/tomography.csv",debug=True)
# result_function(df, resource, distanceMethod='euclidean', outputFormatJson=False, DEBUG_MODE=False)


df = pd.read_csv("C:/Users/ay-admin/PycharmProjects/da4rdm-recsys-cb/Data/tomography_dataframe_new_selectedFeature.csv")
df.drop(columns=["Unnamed: 0"], inplace=True)
evaluate(df)
print("########### DONE ###########")
