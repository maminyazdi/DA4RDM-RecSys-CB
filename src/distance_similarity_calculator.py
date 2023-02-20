""" Comparing Resources within a csv-File

Frameworks required:
    -numpy
    -pandas
    -sklearn

How to use:
    call the method result_function
"""

# Import standard modules
from collections import defaultdict
import copy

# 3rd party modules
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale
import pandas as pd


def calculateCentroids(df, nearestNeighbour, DEBUG_MODE=False):
    """ calculate centroids of a Resource

    :param df: dataframe with all resources
    :param nearestNeighbour: if true use knn else mean of a resource
    :param DEBUG_MODE: debug mode on or off
    :return: a dictionary with all the resource id as a key and the centroid as the value
    """
    if(DEBUG_MODE):
        print('-'*10+'Start calculating centroids'+'-'*10)

    tempDf = df.copy()
    emptyDf = pd.DataFrame()
    if(nearestNeighbour):
        # calculate the centroid with the k-mean algorithm
        n_cluster = 1
        resources = tempDf.Resource.unique()
        # centroidDict = defaultdict()

        # calculate the centroid with the k-mean algorithm
        for resource in resources:
            X = tempDf[tempDf.Resource == resource].iloc[:, 1:]
            kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(X)
            centroid = pd.DataFrame(kmeans.cluster_centers_[n_cluster - 1]).transpose()
            emptyDf = pd.concat([emptyDf, centroid], ignore_index=True)

        if(DEBUG_MODE):
            # print('Centroids:\n', centroidDict)
            # pd.DataFrame.from_dict(centroidDict).to_csv('centroids.csv')
            print('-'*10+'End calculating centroids'+10*'-')

        emptyDf.set_index(pd.Index(resources), inplace=True)
        return emptyDf

    else:
        #calculate centroid by creating the mean
        return tempDf.groupby('Resource').mean()


def getDistanceFunction(distanceMethod: str):
    """ Import distance method from sklearn

    :param distanceMethod: string distanceMethod: 'euclidean' or 'cosine'
    :return: distanceFunction
    """
    if distanceMethod == 'euclidean':
        from sklearn.metrics.pairwise import euclidean_distances
        return euclidean_distances
    elif distanceMethod == 'cosine':
        from sklearn.metrics.pairwise import cosine_distances
        return cosine_distances


def calculateDistances(centroidDataframe, distanceMethod, key: str, sortAscending=True, DEBUG_MODE=False):
    """Calculate distance between the key-resource-centroid and the resources in the centroidDictionary

    :param centroidDataframe: a df with all the centroids to consider
    :param distanceMethod: distanceMethod, 'cosine' or 'euclidean' distance
    :param key: key (type: str) to consider
    :param sortAscending: True = sort df ascending; False = descending
    :param DEBUG_MODE: debug mode on or off
    :return: an ordered dataframe with the distances to the key and scaled between 0 and 1
    """
    if (DEBUG_MODE):
        print('-' * 10 + 'Start calculating distances between centroids' + '-' * 10)

    distanceFunction = getDistanceFunction(distanceMethod)

    #calculate distances between centroids
    distancesDict = {}
    for centroid in centroidDataframe.index:
        # get distances between key-resource and the resources in the dictionary
            distancesDict[centroid] = distanceFunction(centroidDataframe.loc[key].values.reshape(1, -1), centroidDataframe.loc[centroid].values.reshape(1, -1))[0]

    # create a Dataframe with the distances
    distancesDf = pd.DataFrame(distancesDict.values(), index=distancesDict.keys())
    distancesDf.rename(columns={0: 'distance'}, inplace=True)
    if(DEBUG_MODE):
        print('distancesDf', '\033[1m','before', '\033[0m', 'minmaxScaling and sorting:')
        distancesDf.to_csv('distances.csv')
        print(distancesDf)

    #MinMax-Scale the distances between 0 and 1
    #if minmaxScaling and distanceMethod!= 'cosine':
    distancesDf.distance = minmax_scale(distancesDf.distance, feature_range=(0, 1))
    if(DEBUG_MODE):
        distancesDf.to_csv('distancesMinMaxScaled.csv')

    distancesDf.sort_values(by='distance', ascending=sortAscending, inplace=True)

    if(DEBUG_MODE):
        print('distancesDf', '\033[1m', 'after', '\033[0m', 'minmaxScaling and sorting:')
        print(distancesDf)

    if (DEBUG_MODE):
        print('-' * 10 + 'End calculating distances between centroids' + '-' * 10)

    return distancesDf


def result_function(df, key:str, distanceMethod='euclidean', sortAscending=True, nearestNeighbourFlag=True, outputFormatJson=False, DEBUG_MODE=False):
    """Calculating a distance between the key-resource and the resources in the dataframe

    :param df: preprocessed dataframe
    :param key: compare resources to this key
    :param distanceMethod: 'euclidean' or 'cosine' distance
    :param sortAscending: True = sort output ascending; False = descending
    :param nearestNeighbourFlag: Using a different implementation of the calculating the centroids, but the results are the same
    :param outputFormatJson: Trigger Json format
    :param DEBUG_MODE: debug mode
    :return: (relative) distance between key and all resources
    """

    if(DEBUG_MODE):
        print('='*10+'Start'+'='*10)

    tempDf = df.copy(deep=True)

    #calulating centroids with k-means
    centroidDictionary = calculateCentroids(tempDf, nearestNeighbour=nearestNeighbourFlag, DEBUG_MODE=DEBUG_MODE)

    #calculating distance between the key-resource-centroid and all resources
    distancesDf = calculateDistances(centroidDictionary, distanceMethod, key, sortAscending, DEBUG_MODE=DEBUG_MODE)

    if DEBUG_MODE:
        print('='*10+'End'+'='*10)

    if(outputFormatJson):
      return distancesDf.to_json()
    else:
      return distancesDf