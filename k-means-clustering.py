# Lia Ristiana
# second step is to cluster the stratified dataset
# then we write it to csv file

import os

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

import time
start_time = time.time()

input_file="results/sampled/2017-04-02 15:25:22/dataset.csv"

df = pd.read_csv(input_file)

x = df.ix[:,[0,1,2,3,4,5,6]]
y = df.ix[:,7]

k=3

model = KMeans(n_clusters=k)
model.fit(x)

centroids = model.cluster_centers_
labels = model.labels_

# INSERT THE LABEL OF CLUSTER TO DATAFRAME
label_col = pd.Series(labels)
x["class"] = y
x["cluster"] = label_col.values

# SAVE THE WHOLE DATA TO CSV
# csv_name = time.strftime("%Y-%m-%d %H:%M:%S")
# os.makedirs("results/clustered/%s"%csv_name, exist_ok=True)
# x.to_csv(("results/clustered/%s/clustered-dataset.csv"%csv_name), sep=",", encoding="utf-8", index=False)

# SAVE THE DATA TO ITS OWN CSV BASED ON ITS CLUSTER
clusterNames = list(range(0,k))

csv_name = time.strftime("%Y-%m-%d %H:%M:%S")
os.makedirs("results/clustered/%s"%csv_name, exist_ok=True)
for name in clusterNames:
    temp = x.loc[x['cluster']==name]
    temp.to_csv("results/clustered/%s/%s.csv"%(csv_name,name), index=False)



time_elapsed = time.time() - start_time
print("--- %s seconds ---" % (time_elapsed))
