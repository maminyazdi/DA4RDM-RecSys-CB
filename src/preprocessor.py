""" Preprocessor for comparing Resources within a csv-File
Frameworks required:
    -numpy
    -pandas
    -sklearn
    -tensorflow
    -tensorflow-hub
    -tensorflow-text
    -nlpaug
How to use:
 call the method: loadAndPreprocess_function
"""

# Standard library imports
import copy
from enum import IntFlag
import gc
import os
import re


# 3rd party libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as naf
from nlpaug.util import Action
# from imputation import imputatedataframe
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

# Variables
preprocess_url = "https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3"


# changes the string values in the dataframe
def changeStringValuesInDataFrame(df, resourceStringAdd, action):
    testDf = df.copy()

    # method to change string randomly
    def changeString(string):
        if (type(string) != (float or int or complex)):
            if (action == 'substitute'):
                aug = nac.RandomCharAug(action="substitute")
                return aug.augment(string)[0]
            elif (action == 'keyboard'):
                aug = nac.KeyboardAug()
                return aug.augment(string)[0]
            # elif(action == 'contextualWordSubstitute'):
            #   aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")
            #   return aug.augment(string)
            else:
                raise Exception('no valid action value')

    combinedString = action + ' ' + resourceStringAdd + ' ' + '{}'
    # change Resource
    testDf.Resource = testDf.Resource.map(combinedString.format)

    for i in range(0, len(testDf.columns)):
        if (testDf.iloc[:, i].dtype == object and not re.compile('resource|type|file', re.IGNORECASE).match(
                testDf.columns[i])):
            testDf.iloc[:, i] = testDf.iloc[:, i].apply(changeString)

    return testDf


def augmentFile(filepath: str, input_seperator='|', output_seperator='|', debug=False):
    """ Augment file, add more resources by copying a DataFrame, changing the strings in the DataFrame and merging the original and the changed copy DataFrame together.

    :param filepath: filepath to the file
    :param input_seperator: seperator of the input file
    :param output_seperator: seperator of the output file
    :param debug: debug on/off
    :return: filepath to the augmented file
    """
    df = pd.read_csv(filepath, delimiter=input_seperator)
    # remove Url's in header
    df.set_axis(removeURLInHeader(df.columns, debug=debug), axis=1, inplace=True)  # todo: in neue methode auslagern

    # removes url from column 'File'
    if ('File' in df.columns):
        df['File'] = removeUrlInColumnFile(df['File'], debug=debug)

    concatDf = pd.concat([df, changeStringValuesInDataFrame(df, 'origin from', 'keyboard'),
                          changeStringValuesInDataFrame(df, 'origin from',
                                                        'substitute')])  # , changeStringValuesInDataFrame(df, 'origin from', 'substitute')])
    # testDf = changeStringValuesInDataFrame(df, 'origin from', 'keyboard')
    concatDf.to_csv('augmentedFile.csv', sep=output_seperator, index=False)
    return 'augmentedFile.csv'


# fill nan
def fillNan(filepath: str, seperator='|', DEBUG_MODE=False):
    """Fill all nan values in the csv file and create a temporary csv-file.
    Columns of type np.number are filled with 0 and columns of type object with 'None'.

    :param filepath: filepath
    :param seperator: sep to load the csv-file
    :param DEBUG_MODE: debug on/off
    :return: Nothing
    """

    df = pd.read_csv(filepath, sep=seperator)
    # test if nan-values exist in column Resource
    if (df['Resource'].isnull().values.any()):
        raise ValueError('There is a resource without ID')

    drop_threshold = 0.7
    len_dataset = len(df.index)
    for column in df:
        nan_count = df[column].isna().sum()
        nan_count_avg = nan_count / len_dataset
        if nan_count_avg > drop_threshold:
            df = df.drop(columns=[column])

    objectImputer = SimpleImputer(strategy='most_frequent')
    text_columns = df.select_dtypes(include=object).iloc[:, :].columns
    df[text_columns] = pd.DataFrame(objectImputer.fit_transform(df[text_columns]), columns=text_columns)


    numeric_columns = df.select_dtypes(include=np.number).iloc[:, :].columns
    numerical_imputer = KNNImputer(n_neighbors=3)
    df[numeric_columns] = pd.DataFrame(numerical_imputer.fit_transform(df[numeric_columns]), columns=numeric_columns)




    # test if there are any values of type nan
    if (df.isnull().values.any()):
        raise Exception('Nan in csv-file')

    df.to_csv(filepath + 'filled_', index=False)

    if (DEBUG_MODE):
        print("create csv-File:\t filledDataframe.csv\n")
        df.to_csv('filledDataframe.csv')
    return


def removeURLInHeader(columns, debug=False):
    """ removes the url 'http://purl.org/coscine/terms/' in the header

    :param columns: columns of the dataframe
    :param debug: debug on/off
    :return: columns without the url: 'http://purl.org/coscine/terms/'
    """

    if (debug):
        print("-" * 6 + "Start Removing Url in header" + "-" * 6)

    # regex-expression to identify the
    perlUrlString = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+#|http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+terms/"
    # remove the url
    columns = columns.str.replace(perlUrlString, "", regex=True)

    if (debug):
        print("-" * 14 + "End removing Url in header" + "-" * 14 + "\n")
    return columns


def printAction(actionString, value, debug=False):
    """function to log actions in

    :param actionString: performed action
    :param value: changed value
    :param debug: True/False
    :return:
    """
    if (debug):
        print('\033[1m' + actionString + '\033[0m' + value)
    return value


class frameworkFlag(IntFlag):
    nltkFramework = 1
    SpaCyFramework = 2
    tensorflowFramework = 4


def textPreprocessingPipeline(value, encoder, preprocessor, parseLowerCase=True, removePunctuations=True,
                              removeWhitespace=True,
                              tokenize=True, vectorization=True, framework=4, debug=False):
    """ converts a string into a vector

    :param value: string to be vectorized
    :param encoder: encoder to use if mBert is getting used
    :param preprocessor: mBert preprocessor
    :param parseLowerCase: True/False
    :param removePunctuations: True/False
    :param removeWhitespace: True/False
    :param tokenize: True/False
    :param vectorization: True/False
    :param framework: True/False
    :param debug: True/False
    :return: vector representation of a string
    """

    if (type(value) == (float or int or complex)):
        # return value
        value = str(value)

    if (debug):
        print("-" * 14 + "Starting Converting" + "-" * 14)

    value = textpreprocessForVectorization(copy.deepcopy(value), preprocessor, parseLowerCase, removePunctuations,
                                           removeWhitespace,
                                           tokenize, vectorization, framework, debug=debug)

    if (vectorization):
        value = vectorize(copy.deepcopy(value), framework, encoder, debug=debug)

    else:
        if (debug):
            print("Value isn't a string -> no changes")

    if (debug):
        print("-" * 22 + "END" + "-" * 22)
    return value


def vectorize(value, framework, encoder, debug=False):
    if (isinstance(value, str) == False and isinstance(value, dict) == False):
        return value

    if (debug):
        print("-" * 10 + 'Vectorize' + 10 * '-')
    if (framework & frameworkFlag.tensorflowFramework):
        if (debug):
            print("Vectorization with tensorflow")
        outputs = encoder(value)
        if (len(outputs["pooled_output"].numpy()) == 1):
            if (debug):
                printAction('After vectorization:\t', str(outputs["pooled_output"].numpy()[0]), debug=debug)
            if (debug):
                print('-' * 10 + "end vectorization with tensorflow" + '-' * 10)
            return outputs["pooled_output"].numpy()[0]
        else:
            print(len(outputs["pooled_output"].numpy()[0]))
            raise Exception("Length of vector is not 1")
    else:
        raise Exception('Should not get here')


def textpreprocessForVectorization(value, preprocessor, parseLowerCase, removePunctuations, removeWhitespace, tokenize,
                                   vectorization,
                                   framework, debug=False):
    if (debug):
        printAction("Beginning \t\t", value, debug)

    if (type(value) != str):
        value = str(value)

    if (framework & frameworkFlag.tensorflowFramework):
        value = preprocessor([value])
        if (debug):
            printAction("After preprocessing:\n", str(value), debug)
        return value


def assignDataframeToProcessedDataframe(df, solDf, iterator_index):
    tempDf = df.copy()
    if (iterator_index == 0):
        return tempDf
    else:
        return pd.concat([solDf, tempDf], ignore_index=True)


def textPipelineFile(file, chunkSize=318, features=[], encoder='', debug=False):
    """ load and preprocesses the strings in the file

    :param file: path
    :param chunkSize: amount of rows loaded
    :param features: features to select
    :param debug: True/False
    :return: df with preprocessed text
    """
    if (os.path.isfile(file)):
        if (debug):
            print("=" * 20 + "Start Processing File" + "=" * 20 + '\n')
        # filename = copy.deepcopy(file.split('.')[0])
        solDf = pd.DataFrame()
        dataIterator = pd.read_csv(file, chunksize=chunkSize)
        iterator_index = 0
        for df in dataIterator:
            df = processChunk(df.copy(deep=True), features=features, encoder=encoder,
                              debug=debug)

            solDf = assignDataframeToProcessedDataframe(df, solDf, iterator_index)
            iterator_index += 1

        if (debug):
            print("=" * 35 + "End pipelining " + "=" * 35)
        return (solDf)

    else:
        print(file + "does not exist")
    gc.collect()


def manualFeatureSelection(filepath, features, DEBUG_MODE=False):
    """select features manually based on the features provided in the variable features. The other features are dropped.

    :param filepath: path to the dataframe
    :param features: features to select
    :param DEBUG_MODE: sets debug mode
    :return: df with selected features
  """
    df = pd.read_csv(filepath)

    # test if featureList is empty --> return
    if (len(features) == 0):
        if (DEBUG_MODE):
            print('-' * 10 + 'No features selected' + 10 * '-')
        return df
    # get column names from df
    dropList = df.columns.to_list()
    # remove every feature in the drop list + Resource
    dropList.remove('Resource')
    for featureName in features:
        if featureName in dropList:
            if (DEBUG_MODE):
                print('Keep feature:\t', featureName)
            dropList.remove(featureName)
        elif (featureName is not 'Resource'):
            raise ValueError(f"feature {featureName} does not exist or is specified multiple times")
    # drop columns with names list in the df
    if (DEBUG_MODE):
        print('-' * 10 + 'drop columns:' + 10 * '-')
        print(dropList)

    # test if there is at least one viable features
    if ((len(df.columns.to_list()) - len(dropList)) <= 1):
        if (DEBUG_MODE):
            print('not viable features selected')
        raise Exception(f"No viable feature selected. Features selected:\t{features}")

    df.drop(columns=dropList, axis=1, inplace=True)

    os.remove(filepath)
    df.to_csv(filepath, index=False)


def processChunk(df, features, encoder, debug=False):
    """ preprocesses the text of a chunk of the dataframe

    :param df: dataframe
    :param features: features to select
    :param debug: True/False
    :return: df with preprocessed text
    """

    # remove Url's in header
    df.set_axis(removeURLInHeader(df.columns, debug=debug), axis=1, inplace=True)

    # removes url from column 'File'
    if ('File' in df.columns):
        df['File'] = removeUrlInColumnFile(df['File'], debug=debug)

    if (debug):
        print("drop columns File and Type")
    df.drop(columns=['File', 'Type'], inplace=True, errors='ignore')

    df = textProcessing(df.copy(), encoder=encoder, debug=debug)
    return df


def removeUrlInColumnFile(series, debug):
    """Removes the URL in column 'File'. The URL gets removed until 'path=%'

    :param series: column 'File'
    :param debug: True/False
    :return: series without the hdl-URL-String
    """

    # removes an URL until the part with path=%
    hdlURLString = "http[s]?://(hdl.handle.net/)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+?(path=%)"  # 'https://regexr.com
    if (debug):
        print("-" * 10 + "Remove Url in column File" + 10 * "-")
        print('Before:\t' + series.iloc[0])
    tempSeries = series.str.replace(hdlURLString, "", regex=True)

    if (debug):
        print('After\t' + tempSeries)
        print("-" * 10 + "End removing Url from File column" + "-" * 10 + "\n")
    return tempSeries


def textProcessing(df, parseLowerCase=True, removePunctuations=True, removeWhitespace=True,
                   tokenize=True, vectorization=True, framework=4, encoder='', debug=False):
    """ methods to preprocess the text

    :param df: dataframe
    :param parseLowerCase: True/False
    :param removePunctuations: True/False
    :param removeWhitespace: True/False
    :param tokenize: True/False
    :param vectorization: True/False
    :param framework: Flag to set the framework: nltk = 1, SpaCy = 2, tensorflow = 4
    :param debug: True/False
    :return: df with text preprocessed
    """

    encoder = getEncoder(encoder)
    # load model
    preprocessor = hub.KerasLayer(preprocess_url)

    for i in range(len(df.columns)):
        # ignore columns: Resource, Type, File
        if (re.compile('resource|type|file', re.IGNORECASE).match(df.columns[i])):
            if (debug):
                print("\nDoesn't pipeline column " + df.columns[i])
        # on the other columns:
        elif (df.iloc[:, i].dtype == object):
            if (debug):
                print('\t' + df.columns[i])
            # apply the text preprocessing pipeline
            df.iloc[:, i] = df.iloc[:, i].copy().apply(textPreprocessingPipeline, args=(
                encoder, preprocessor, parseLowerCase, removePunctuations, removeWhitespace,
                tokenize, vectorization, framework, debug))
            gc.collect()
        else:
            if (debug):
                print('Column: ', df.columns[i], ' does not need to get processed')
    return df


def getEncoder(encoder):
    encoder_url = ''
    if (encoder == 'mobilebert_multi_cased'):
        encoder_url = 'https://tfhub.dev/tensorflow/mobilebert_multi_cased_L-24_H-128_B-512_A-4_F-4_OPT/1'
    elif (encoder == 'bert_multi_cased'):
        encoder_url = 'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4'
    else:
        raise Exception('Wrong encoder string')
    # set model for text preprocessing
    encoder = hub.KerasLayer(
        encoder_url,
        trainable=False
    )
    return encoder


# def pcaOnSeries(series, n_componentsAfterPCA = 1, DEBUG_MODE=False):
#     """ Applies PCA on a Series. The new length is n_componentsAfterPCA.

#     :param series: pd.series
#     :param DEBUG_MODE: True/False
#     :return: reduced series to an array of length 1
#     """

#     from sklearn.decomposition import PCA
#     pca = PCA(n_components=n_componentsAfterPCA)

#     # applies pca on a series of object
#     if (series.dtype == object):
#         if(DEBUG_MODE):
#             print('Start PCA')

#         #formating the series into an array for the pca
#         tempSeries = copy.copy(series)
#         # for i in range(len(series)):
#         tempArray = []
#         for vector in tempSeries:
#             tempArray.append(vector)
#             # tempArray = series.values

#         lda_components = pca.fit_transform(tempArray)
#         if(DEBUG_MODE):
#             print('END PCA')
#         return lda_components
#     else:
#         # return series
#         print('not object type')


# # def transformSeriesIntoArray(series):


def pcaOnDataframe(df, features, n_componentsAfterPCA, debug=False):
    """ Applies PCA onto every column of type object in the dataframe

    :param df: dataframe
    :param debug: true/false
    :return: reduced dataframe
    """

    if n_componentsAfterPCA < 0:
        return df
    if (features == []):
        if debug:
            printAction('Start PCA on the dataFrame', '', debug)

        ################    code from an earlier version    ##############
        #     #if n_components after pca is >=1 apply pca
        # #    if(n_componentsAfterPCA>=1):
        #     #Applies PCA onto every column of type object in the dataframe
        #     for i in range(len(tempDf.columns)):
        #       if (tempDf.iloc[:, i].dtype == object and tempDf.columns[i] != 'Resource'):
        #           if(debug):
        #               print('PCA on column: ', tempDf.columns[i])
        #           tempSeries = pcaOnSeries(tempDf.iloc[:, i], n_componentsAfterPCA=n_componentsAfterPCA, DEBUG_MODE=debug)

        #     #              #add the values of the series to the dataset
        #     #              for n in range(n_componentsAfterPCA):
        #     #                tempDf[tempDf.columns[i]+str(n)] = tempSeries[:, n]

        #     # not applying pca, but the features values of a feature need to be put into a column otherwise we are getting errors
        # #    if(n_componentsAfterPCA==-1):
        ########################################################

        from sklearn.decomposition import PCA
        tempDf = df.copy()

        # deciding on the vector length after the pca
        if (n_componentsAfterPCA == 0):
            # pca is automatically deciding the vector length after the pca
            pca = PCA()
        elif (n_componentsAfterPCA > 0):
            # choosing manual vector length after the pca
            pca = PCA(n_components=n_componentsAfterPCA)

        # applying pca, except on the 'Resource'(-id) column
        lda_components = pca.fit_transform(tempDf.iloc[:, 1:])

        # drop all columns except for 'Resource'
        tempDf.drop(df.columns.drop('Resource'), axis=1, inplace=True)

        # assign values to dataFrame
        tempDf = tempDf.join(pd.DataFrame(lda_components))

        if debug:
            printAction('End PCA on the dataFrame', '', debug)
        return tempDf
    else:
        return df


def splitVectorsIntoNewColumn(debug, resourceSeries, tempDf):
    # drill dimension of features and project into dataframe
    # iterate all columns of type object except of Resource, because it's not a vector
    for i in range(len(tempDf.columns)):
        if (tempDf.iloc[:, i].dtype == object and tempDf.columns[i] != 'Resource'):
            # apply on the series: split array into columns in the tempDf
            tempSeries = copy.copy(tempDf.iloc[:, i])
            tempArray = []

            for vector in tempSeries:
                tempArray.append(vector)

            # create names of the columns
            columnNames = [tempDf.columns[i] + str(x) for x in range(0, len(tempArray[0]))]
            # create dataframe to parse join data
            transferDataFrame = pd.DataFrame(tempArray, columns=columnNames)
            # copy data into tempDf
            for column in transferDataFrame:
                tempDf[column] = transferDataFrame[column]
    # removing feature vectors in one column by deleting all columns of type object, because there redundant
    if (debug):
        printAction('Remove columns not of type number', '')
    tempDf = tempDf.select_dtypes(include=np.number)
    # add Resource again because it got deleted as well
    tempDf['Resource'] = resourceSeries.copy()
    # Move last column(Resource) to the first to avoid errors
    tempDf2 = pd.DataFrame(tempDf)
    temp_cols = tempDf2.columns.tolist()
    new_cols = temp_cols[-1:] + temp_cols[:-1]
    tempDf2 = tempDf2[new_cols]
    return tempDf2


# source: https://stackoverflow.com/questions/54405704/check-if-all-values-in-dataframe-column-are-the-same
def is_unique(s) -> bool:
    """Tests if there is only 1 value in series

    :param s: column as a pandas series
    :return: if there is only 1 value in the series
    """
    a = s.to_numpy()  # s.values (pandas<0.24)
    return (a[0] == a).all()


def removeColumnsWithOnlyOneValue(filepath: str, DEBUG_MODE=False):
    """ removes a column if the columns only has one value(0 Variance). It replaces the original csv-File.

    :param filepath: path to the file
    :param DEBUG_MODE: debug mode on/off
    :return: None
    """
    if (DEBUG_MODE):
        print('removeColumnsWith1Value')
    tempDf = pd.read_csv(filepath)
    columns = tempDf.columns
    # remove every column which has only 1 value
    for column in columns:
        if (is_unique(tempDf[column])):
            if (DEBUG_MODE):
                print(f"Drop column:\t{column}")
            tempDf.drop(labels=column, axis=1, inplace=True)
    os.remove(filepath)
    tempDf.to_csv(filepath, index=False)
    return


def testInputs(filepath: str, features, seperator: str, scalingFlag, DEBUG_MODE=False):
    """Tests if the specified inputs are viable. If they are not, it raises an Exception.

    :param filepath: filepath to the csv-file
    :param features: array of features
    :param seperator: seperator to read the csv-file
    :param scalingFlag: scalingFlag
    :param DEBUG_MODE:
    :return: If inputs are viable => True
    """
    if (os.path.exists(filepath) == False):
        raise Exception(f"File in path {filepath} does not exist")
    df = pd.read_csv(filepath, sep=seperator)
    # test amount of rows >2
    if (df.shape[1] <= 2):
        raise Exception(f'Dataframe has only {df.shape[1] - 1} row, but it needs at least 2')
    # test if the scalingFlag is valid
    testScalingFlag(scalingFlag)
    return True


def testScalingFlag(scalingFlag):
    scalingFlags = ['none', 'minmax_scale', 'maxabs_scale',
                    'normalizeSample', 'normalizeFeatures', 'standard_scale', 'robust_scale']
    if scalingFlag in scalingFlags:
        return
    else:
        raise Exception(f'false scalingflag:\t{scalingFlag}')


def scaleDataframe(debug, df, scalingFlag):
    """ scales dataFrame

    :param debug: debug mode on/off
    :param df: dataFrame to scale
    :param scalingFlag: 'none', 'minmax_scale', 'maxabs_scale', 'normalizeSample', 'normalizeFeatures', 'standard_scale', 'robust_scale'
    :return: return scaled dataframe, except 'none' is chosen as the scalingFlag
    """

    if (debug):
        printAction('Start scaling', '', False)

    if (scalingFlag == 'none'):
        if (debug):
            printAction('Not scaling', 'df', debug)
        return df.iloc[:, 1:]
    tempDf = df.copy()
    if (scalingFlag == 'minmax_scale'):
        if (debug):
            printAction('minmax_scale', 'values of the dataframe')
        from sklearn.preprocessing import minmax_scale
        return minmax_scale(tempDf.iloc[:, 1:])
    if (scalingFlag == 'maxabs_scale'):
        # scales all features to the values [-1, 1]
        if (debug):
            printAction('maxabs_scale', 'values of the dataframe', debug=debug)
        from sklearn.preprocessing import maxabs_scale
        return maxabs_scale(tempDf.iloc[:, 1:])
    if (scalingFlag == 'normalizeSample'):
        if (debug):
            printAction('normalize samples', 'values of the dataframe', debug=debug)
        from sklearn.preprocessing import normalize
        return normalize(tempDf.iloc[:, 1:], axis=1, copy=False)
    if (scalingFlag == 'normalizeFeatures'):
        if (debug):
            printAction('normalize features', 'values of the dataframe', debug=debug)
        from sklearn.preprocessing import normalize
        return normalize(tempDf.iloc[:, 1:], axis=0, copy=False)
    if (scalingFlag == 'standard_scale'):
        if (debug):
            printAction('standard scale features', 'values of the dataframe', debug=debug)
        from sklearn.preprocessing import scale
        return scale(tempDf.iloc[:, 1:])
    if scalingFlag == 'robust_scale':
        if (debug):
            printAction('robustScaling features', 'values of the dataframe', debug=debug)
        from sklearn.preprocessing import robust_scale
        return robust_scale(tempDf.iloc[:, 1:])

    raise Exception("No valid scalingFlag")


def removeOutliers(df, threshold=3, DEBUG_MODE=True):
    """ removing outliers based on distance between centroid, relative to their z-score

    :param df:
    :param threshold: threshold for the z-score
    :param DEBUG_MODE: debug on/off
    :return: dataframe with outliers filtered
    """
    if threshold <= 0:
        return df
    from sklearn.metrics import pairwise_distances
    tempDf = df.copy()

    # avarage distance
    tempDf = tempDf.groupby('Resource').mean()
    pairwiseDistances = pairwise_distances(tempDf)
    avarageDistances = pd.Series([sum(array) / len(array) for array in pairwiseDistances], index=tempDf.index)

    # calculate z-Score for every resource
    from scipy.stats import zscore
    zscoreAvarageDistances = zscore(avarageDistances)
    if DEBUG_MODE:
        print('zscoreAvarageDistances:\t')
        print(zscoreAvarageDistances)

    outlier = zscoreAvarageDistances[zscoreAvarageDistances > threshold]
    if DEBUG_MODE:
        print('outlier\t')
        print(outlier)

    # return if there are no outlier
    if outlier.empty:
        return df

    # remove outliers from dataframe
    removedOutlierDf = df.copy()
    for resource in outlier.index:
        removedOutlierDf = removedOutlierDf[removedOutlierDf['Resource'] != resource]

    return removedOutlierDf


def loadAndPreprocess_function(filepath: str, features=[], chunkSize=318, n_componentsAfterPCA=0, seperator='|',
                               encoder='mobilebert_multi_cased', zScaleThreshold=3, scalingFlag='maxabs_scale',
                               debug=False):
    """Loads and preprocesses a csv-file

    :param filepath: filepath to the csv file with '|' as the seperator
    :param features: array of features to consider
    :param chunkSize: amount of rows to process with nlp
    :param n_componentsAfterPCA: vector length after pca, if <0 --> without pca, if ==0 automatic vector length
    :param seperator: the seperator used when loading the csv-file
    :param encoder: set the language model: 'mobilebert_multi_cased' or 'bert_multi_cased'
    :param zScaleThreshold: z-scale threshold to filter outliers
    :param scalingFlag: minmax scaling resource vector
    :param debug: debug mode
    :return: a preprocessed pandas.DataFrame
    """
    if (debug):
        print('=' * 5, 'Start preprocessingAndLoad_function', '=' * 5)
    # test the if the inputs are viable, throws error if not
    if (testInputs(filepath, features, seperator, scalingFlag, DEBUG_MODE=debug) == True):

        fillNan(filepath, seperator, DEBUG_MODE=debug)

        if (features != []):
            # manuell features selection
            manualFeatureSelection(filepath + 'filled_', features, DEBUG_MODE=debug)
        #        else:#-->should not interfere with users decision to select features
        #            #automatic features selection, by removing columns with only one value
        #            if(removeColumnsWithOneValueFlag):
        #                removeColumnsWithOnlyOneValue(filepath + 'filled_', DEBUG_MODE=debug)

        df = textPipelineFile(filepath + 'filled_', chunkSize=chunkSize, features=features, encoder=encoder,
                              debug=debug)
        if (debug):
            print("create csv file from dataframe after text processing")
            df.to_csv(filepath + 'afterTextProcessing.csv', index=False)
        os.remove(filepath + 'filled_')

        # split column-vector values into new columns for each value
        df = splitVectorsIntoNewColumn(debug, df['Resource'], df)

        # drop outliers
        df = removeOutliers(df, zScaleThreshold, debug)

        # pca on dataframe
        df = pcaOnDataframe(df, features, n_componentsAfterPCA, debug=debug)
        if (debug):
            print("create csv file from dataframe after pca")
            df.to_csv(filepath + 'afterPCA' + str(n_componentsAfterPCA) + ".csv", index=False)

        df.iloc[:, 1:] = scaleDataframe(debug, df, scalingFlag)

        if (debug):
            print("create csv file from dataframe after minmax-scaling")
            df.to_csv(filepath + 'afterTextMinMaxScaling.csv', index=False)
            print('=' * 10, 'END', '=' * 10)
        return df
