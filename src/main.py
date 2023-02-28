import pandas as pd

from preprocessor import loadAndPreprocess_function
from distance_similarity_calculator import result_function
from evaluation import evaluate
# df = loadAndPreprocess_function(filepath="Data/tomography.csv", features=['http://purl.org/coscine/terms/sfb1394#acquiredIons', 'http://purl.org/coscine/terms/sfb1394#annularMillingParameters', 'http://purl.org/coscine/terms/sfb1394#baseTemperature', 'http://purl.org/coscine/terms/sfb1394#laserPulseEnergy', 'http://purl.org/coscine/terms/sfb1394#lowVoltageCleaning', 'http://purl.org/coscine/terms/sfb1394#pulseFrequency','http://purl.org/coscine/terms/sfb1394#runTime','http://purl.org/coscine/terms/sfb1394#specimenApexRadius'],debug=False)
# result_function(df, resource, distanceMethod='euclidean', outputFormatJson=False, DEBUG_MODE=False)

df = pd.read_csv("DataProcessed.csv")
df.drop(columns=["Unnamed: 0"], inplace=True)
evaluate(df)
print("Output")
