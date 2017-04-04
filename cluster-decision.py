# Lia Ristiana
# next step is to decide the cluster where the testing data belongs to
# by calculating the euclidean distance from the testing datum to the centroid of each cluster
# then assign the datum to the cluster where its distance to the centroid is closest to.
# Finally, store the data to its own file based where it belongs in a cluster.

import csv
import os

import pandas as pd
import numpy as np

from scipy.spatial import distance

import time
start_time = time.time()

# FETCH THE TESTING SET
dir_name = "2017-04-04 15:38:24"
input_file="results/testing/%s/testing-set.csv"%(dir_name)
df_testing = pd.read_csv(input_file)

# FETCH THE CENTROIDS
dir_name = "2017-04-05 00:38:39"
input_file="results/clustered/%s/centroids.csv"%dir_name
df_centroids= pd.read_csv(input_file)

centroids = df_centroids.as_matrix()

# for i in range(len(df_testing)):
#     for j in range(len(df_centroids)):


distance_matrix = []
for index_t, row_t in df_testing.ix[:,[0,1,2,3,4,5,6]].iterrows():
    dist = []
    # for index_c, row_c in df_centroids.iterrows():
    dst = 1
    smaller=0
    for i in range(len(centroids)):
        temp = dst
        dst = distance.euclidean(centroids[i], row_t)
        smaller = min(dst,temp)
        
        # print(dst)
    print(index_t,smaller,ind)
    # distance_matrix[index_t] = smaller
    # distance_matrix = distance_matrix.append(distance_matrix)


# print(distance_matrix)
# dst = distance.euclidean(df_testing)
