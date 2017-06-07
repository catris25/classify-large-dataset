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

def cluster_test(training_file, testing_file):
    centroids_file = training_file+"/centroids.csv"
    df_testing = pd.read_csv(testing_file)
    df_centroids= pd.read_csv(centroids_file)

    centroids = df_centroids.as_matrix()

    # Decide which cluster a data point is closest to
    # by using euclidean distance
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

    # Save all data to its own file based on the closest cluster
    # to be used later as a testig set
    count_temp = []

    testing_clustered_dir = "output/testing_clustered/"+str(len(centroids))+"clusters/"
    os.makedirs(testing_clustered_dir, exist_ok=True)
    for i in range(len(centroids)):
        temp = df_testing.loc[df_testing['closest_cluster']==i]
        dir_name = testing_clustered_dir+str(i)
        temp.to_csv(dir_name+".csv", index=False)
        classCount = temp['class'].value_counts(sort=False)
        print(classCount)
    return testing_clustered_dir

def main():
    training_file = "output/training_clustered/5clusters 2017-06-07 23:39:07"
    testing_file = "output/testing_sampled/2017-06-07 23:49:25/testing-set.csv"

    clsr = cluster_test(training_file,testing_file)
    print("result:",clsr)

if __name__ == "__main__":
    main()

# the end
