# Lia Ristiana
# k means clsustering custom code
# second step is to cluster the stratified dataset
# then we write it to csv file

import os

import matplotlib.pyplot as plt
from scipy.spatial import distance

import numpy as np
import pandas as pd

import time
start_time = time.time()

dir_name = "2017-04-13 12:33:41"
input_file="results/sampled/%s/dataset.csv"%dir_name

df = pd.read_csv(input_file)

x = df.ix[:,[0,1,2,3,4,5,6]]
x = x.values
y = df.ix[:,7]

# x_array = np.array([[10.979,7.748],
#             [9.352,8.743],
#             [9.621,7.813],
#             [9.666,12.14],
#             [25.899,11.1],
#             [24.012,8.39],
#             [25.08,9.437],
#             [29.752,10.36]])


class K_Means:
    def __init__(self, k=3, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):

            self.clustering = {}

            for i in range(self.k):
                self.clustering[i] = []

            sum_distances = 0
            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                cluster_label = distances.index(min(distances))
                self.clustering[cluster_label].append(featureset)

                sum_distances += min(distances)

            # get Sum of Squared Errors
            self.sse = sum_distances

            prev_centroids = dict(self.centroids)

            for cluster_label in self.clustering:
                self.centroids[cluster_label] = np.average(self.clustering[cluster_label], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid)/original_centroid*100.0)>self.tol:
                    optimized = False

            if optimized:
                break


clr = K_Means()
clr.fit(x)

# for i in range(len(clr.clustering)):
#     print(i,". ",clr.clustering[i])

# SAVE THE DATA TO ITS OWN CSV BASED ON ITS CLUSTER
clusterNames = list(range(0,k))

dir_name = "%sclusters"%k
os.makedirs("results/clustered/%s"%dir_name, exist_ok=True)

# store the centroids
centroids = pd.DataFrame(clr.centroids)
df_centroids = pd.DataFrame(centroids,columns=['attr1', 'attr2', 'attr3', 'attr4', 'attr5', 'attr6', 'attr7'])
df_centroids.to_csv("results/clustered/%s/centroids.csv"%dir_name, index=False)
print("centroid:\n",centroids)
print("SSE: ",clr.sse)

for name in clustering:
    temp = x.loc[x['cluster']==name]
    temp.to_csv("results/clustered/%s/%s.csv"%(dir_name,name), index=False)
    classCount = temp['class'].value_counts(sort=False)
    print(classCount)

print("The result files are in the %s"%(dir_name))


time_elapsed = time.time() - start_time
print("--- %s seconds ---" % (time_elapsed))
