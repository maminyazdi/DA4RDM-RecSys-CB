from preprocessor import loadAndPreprocess_function
from distance_similarity_calculator import result_function

df = loadAndPreprocess_function(filepath="tomography.csv", features=['http://purl.org/coscine/terms/sfb1394#acquiredIons', 'http://purl.org/coscine/terms/sfb1394#annularMillingParameters', 'http://purl.org/coscine/terms/sfb1394#baseTemperature', 'http://purl.org/coscine/terms/sfb1394#laserPulseEnergy', 'http://purl.org/coscine/terms/sfb1394#lowVoltageCleaning', 'http://purl.org/coscine/terms/sfb1394#pulseFrequency','http://purl.org/coscine/terms/sfb1394#runTime','http://purl.org/coscine/terms/sfb1394#specimenApexRadius'],debug=False)
jsonOutPut = result_function(df, '1EC47F72-DF63-4D95-94E7-EB70C6BA09DB',distanceMethod='euclidean', outputFormatJson=True, DEBUG_MODE=False)
print("Output")
print(jsonOutPut)