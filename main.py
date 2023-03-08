from src.preprocessor import loadAndPreprocess_function
from src.evaluation import evaluate
import pandas as pd

#df = loadAndPreprocess_function(filepath="C:/Users/ay-admin/PycharmProjects/da4rdm-recsys-cb/Data/tomography-new.csv", features=['http://purl.org/coscine/terms/sfb1394#baseTemperature', 'http://purl.org/coscine/terms/sfb1394#laserPulseEnergy', 'http://purl.org/coscine/terms/sfb1394#pulseFrequency',
#'http://purl.org/coscine/terms/sfb1394#acquiredIons', 'http://purl.org/coscine/terms/sfb1394#annularMillingParameters', 'http://purl.org/coscine/terms/sfb1394#detectionRate',  'http://purl.org/coscine/terms/sfb1394#lowVoltageCleaning', 'http://purl.org/coscine/terms/sfb1394#runTime', 'http://purl.org/coscine/terms/sfb1394#shankAngle', 'http://purl.org/coscine/terms/sfb1394#specimenApexRadius', 'http://purl.org/coscine/terms/sfb1394#startVoltage', 'http://purl.org/coscine/terms/sfb1394#stopVoltage'],debug=True)
#df.to_csv('tomography_dataframe_new_few_selectedFeatures.csv')

df = pd.read_csv("C:/Users/ay-admin/PycharmProjects/da4rdm-recsys-cb/src/tomography_dataframe_new_few_selectedFeatures.csv")
df.drop(columns=["Unnamed: 0"], inplace=True)
evaluate(df)
print("########### DONE ###########")
# TODO: Move plot files to be stored in their own folder
# TODO: Add bar plot to view missing values (Data Sparsity) Refer to Markus thesis Figure 6.2a for the selected Features above
# TODO: Add bar plot to demonstrated number of files per resources