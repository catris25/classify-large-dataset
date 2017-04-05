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

closest_cluster = []
for index_t, row_t in df_testing.ix[:,[0,1,2,3,4,5,6]].iterrows():
    dist = []
    for i in range(len(centroids)):
        dst = distance.euclidean(centroids[i], row_t)
        dist.append(dst)

    smallest = min(np.hstack(dist))
    cluster = dist.index(smallest)
    closest_cluster.append(cluster)

df_testing['closest_cluster'] = closest_cluster

count_temp = []
for i in range(len(centroids)):
    temp = df_testing.loc[df_testing['closest_cluster']==i]
    # temp.to_csv("results/try-%s.csv"%i, index=False)
    classCount = temp['class'].value_counts(sort=False)
    print(classCount)
