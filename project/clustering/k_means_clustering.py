# Lia Ristiana
# k means clustering custom code
# second step is to cluster the stratified dataset
# then we write it to csv file

import os

import numpy as np
import pandas as pd

import time

class K_Means:
    def __init__(self, k=k, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            print("iteration: ",i)

            self.clustering = {}

            for i in range(self.k):
                self.clustering[i] = []

            sum_distances = 0
            self.labels =[]
            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                cluster_label = distances.index(min(distances))
                self.labels.append(cluster_label)
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


def kmeans(input_file, k_size):
    df = pd.read_csv(input_file)
    k = k_size

    x = df.ix[:,[0,1,2,3,4,5,6]].values
    y = df.ix[:,7].values

    clr = K_Means()
    clr.fit(x)

    # SAVE THE DATA TO ITS OWN CSV BASED ON ITS CLUSTER
    clusterNames = list(range(0,k))

    dir_name = "%sclusters"%k
    os.makedirs("results/clustered/%s"%dir_name, exist_ok=True)

    # store the centroids
    centroids = pd.DataFrame(clr.centroids)

    centroids = np.transpose(centroids)
    df_centroids = pd.DataFrame(centroids)
    cols=['attr1', 'attr2', 'attr3', 'attr4', 'attr5', 'attr6', 'attr7']
    print(df_centroids)
    df_centroids.to_csv("results/clustered/%s/centroids.csv"%dir_name, index=False)
    print("SSE: ",clr.sse)

    labels = clr.labels
    df["cluster"] = labels

    for name in clusterNames:
        print("name:", name)
        temp = df.loc[df['cluster']==name]
        temp.to_csv("results/clustered/%s/%s.csv"%(dir_name,name), index=False)
        classCount = temp['class'].value_counts(sort=False)
        print(classCount)
        print("total data = %s"%classCount.sum())

    print("The result files are in the %s"%(dir_name))
    clustered_set_dir = "results/clustered/"+dir_name
    return clustered_set_dir

def main():
    dir_name = __main__.main_sampled_dir_name
    input_file="results/sampled/%s/dataset.csv"%dir_name
    print(kmeans(input_file, k_size))


if __name__ == "__main__":
    main()

# the end
