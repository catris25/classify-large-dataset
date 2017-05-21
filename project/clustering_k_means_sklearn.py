# Lia Ristiana
# second step is to cluster the stratified dataset
# then we write it to csv file

import os

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

import time



def clustering(input_file, k_size):
    df = pd.read_csv(input_file)
    x = df.ix[:,[0,1,2,3,4,5,6]]
    y = df.ix[:,7]

    k=k_size

    model = KMeans(n_clusters=k)
    model.fit(x)

    centroids = model.cluster_centers_
    labels = model.labels_

    # INSERT THE LABEL OF CLUSTER TO DATAFRAME
    label_col = pd.Series(labels)
    x["class"] = y
    x["cluster"] = label_col.values

    # SAVE THE DATA TO ITS OWN CSV BASED ON ITS CLUSTER
    clusterNames = list(range(0,k))

    dir_name = "%sclusters"%k
    os.makedirs("output/training_clustered/%s"%dir_name, exist_ok=True)

    # store the centroids
    df_centroids = pd.DataFrame(centroids,columns=['attr1', 'attr2', 'attr3', 'attr4', 'attr5', 'attr6', 'attr7'])
    df_centroids.to_csv("output/training_clustered/%s/centroids.csv"%dir_name, index=False)

    for name in clusterNames:
        temp = x.loc[x['cluster']==name]
        temp.to_csv("output/training_clustered/%s/%s.csv"%(dir_name,name), index=False)
        classCount = temp['class'].value_counts(sort=False)
        print("Cluster-%s"%name)
        print(classCount)

    print("The result files are in the %s"%(dir_name))
    return "output/training_clustered/%s"%dir_name

def main():
    k_size = 22

    input_file="output/training_sampled/2017-05-07 23:12:26/training_set.csv"

    print(clustering(input_file, k_size))


if __name__ == "__main__":
    main()

# the end
