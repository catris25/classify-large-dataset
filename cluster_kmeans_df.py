# Lia Ristiana
# k means clsustering custom code
# second step is to cluster the stratified dataset
# then we write it to csv file

import matplotlib.pyplot as plt
from scipy.spatial import distance

import numpy as np
import pandas as pd

import time
start_time = time.time()

# dir_name = "2017-04-13 12:33:41"
# input_file="results/sampled/%s/dataset.csv"%dir_name
#
# df = pd.read_csv(input_file)
#
# x = df.ix[:,[0,1,2,3,4,5,6]]
# y = df.ix[:,7]

x_array = np.array([[10.979,7.748],
            [9.352,8.743],
            [9.621,7.813],
            [9.666,12.14],
            [25.899,11.1],
            [24.012,8.39],
            [25.08,9.437],
            [29.752,10.36]])

x = pd.DataFrame(x_array)

class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = data.head(self.k)

        # def get_label(featureset):
        #     return np.linalg.norm(self.centroids - featureset, axis=1).argmin()

        for i in range(self.max_iter):

            self.clustering = pd.DataFrame(index=range(0,self.k))

            sum_distances = 0
            for row in data.itertuples(index=False):
                distances = np.linalg.norm(self.centroids - row, axis=1)
                cluster_label = distances.argmin()
                print(distances)
                # self.clustering[cluster_label].append(row)
                # self.clustering.loc[cluster_label].append(row)

                self.clustering.loc[cluster_label]


                # sum_distances += min(distances)

            # for row in data.itertuples(index=False):
            #     index, datum = row
            #     datum = datum.tolist()
            #     print(datum)
            # labels = np.apply_along_axis(get_label, 1, data)
            # print(labels)
            # get Sum of Squared Errors
            self.sse = sum_distances

            # prev_centroids = dict(self.centroids)
            #
            # for cluster_label in self.clustering:
            #     self.centroids[cluster_label] = np.average(self.clustering[cluster_label], axis=0)
            #
            # optimized = True
            #
            # for c in self.centroids:
            #     original_centroid = prev_centroids[c]
            #     current_centroid = self.centroids[c]
            #     if np.sum((current_centroid - original_centroid)/original_centroid*100.0)>self.tol:
            #         optimized = False
            #
            # if optimized:
            #     break

clr = K_Means()
clr.fit(x)

# for i in range(len(clr.clustering)):
#     print(i,". ",clr.clustering[i])
#
# centroids = pd.DataFrame(clr.centroids)
# print("centroid:\n",centroids)
# print("SSE: ",clr.sse)
# print(x)
time_elapsed = time.time() - start_time
print("--- %s seconds ---" % (time_elapsed))
