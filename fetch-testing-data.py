# Lia Ristiana
# next step is to stratify the dataset for sampling
# with fixed number of data from each class
# then we calculate the euclidean distance from each data point
# to the centroid of each cluster that's going to be used as training set.
# We look for the closest centroid and save the data to each own file
# based on the cluster the data point is closest to.

import csv
import os

import pandas as pd
import numpy as np

from scipy.spatial import distance

import time
start_time = time.time()

n_size = 1000

# FETCH THE TESTING SET
input_file="/home/lia/Documents/FINAL-GEMASTIK/training.csv"
df = pd.read_csv(input_file)

# FETCH THE CENTROIDS
dir_name = "5clusters"
input_file="results/clustered/%s/centroids.csv"%dir_name
df_centroids= pd.read_csv(input_file)

centroids = df_centroids.as_matrix()


df['class'] = df['class'].astype('category')

pd.set_option('float_format', '{:f}'.format)

# DEFINE CLASS NAMES AS NEEDED
classNames = [1,2,3,4,5]

df_stratified = pd.DataFrame()

for name in classNames:
	temp = df.loc[df['class']==name].sample(n=n_size)
	df_stratified = pd.concat([df_stratified,temp])

print("RESULT ARRAY")
print(df_stratified.head(5))
classCount = df_stratified['class'].value_counts(sort=False)
print("class frequency\n%s."%classCount)
print("total data = %s"%classCount.sum())

df_testing = df_stratified

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
csv_name = time.strftime("%Y-%m-%d %H:%M:%S")
local_dir = "results/testing-set/"
os.makedirs("%s%s"%(local_dir,csv_name), exist_ok=True)

count_temp = []
for i in range(len(centroids)):
    temp = df_testing.loc[df_testing['closest_cluster']==i]
    temp.to_csv("results/testing-set/%s/%s.csv"%(csv_name,i), index=False)
    classCount = temp['class'].value_counts(sort=False)
    print(classCount)

print("The result files are in the %s"%csv_name)

time_elapsed = time.time() - start_time
print("--- %s seconds ---" % (time_elapsed))
