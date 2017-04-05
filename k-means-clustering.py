# Lia Ristiana
# second step is to cluster the stratified dataset
# then we write it to csv file

import os

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

import time
start_time = time.time()

dir_name = "2017-04-05 23:42:22"
input_file="results/sampled/%s/dataset.csv"%dir_name

df = pd.read_csv(input_file)

x = df.ix[:,[0,1,2,3,4,5,6]]
y = df.ix[:,7]

k=4

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

# store the centroids
print("centroid")
df_centroids = pd.DataFrame(centroids,columns=['attr1', 'attr2', 'attr3', 'attr4', 'attr5', 'attr6', 'attr7'])
df_centroids.to_csv("results/clustered/%s/centroids.csv"%csv_name, index=False)

count_temp = []
for name in clusterNames:
    temp = x.loc[x['cluster']==name]
    temp.to_csv("results/clustered/%s/%s.csv"%(csv_name,name), index=False)
    classCount = temp['class'].value_counts(sort=False)
    print("class frequency\n%s."%classCount)
    print("total data = %s"%classCount.sum())

# classCount = x['class'].value_counts(sort=False)
# print("class frequency\n%s."%classCount)
# print("total data = %s"%classCount.sum())
# classCount.to_csv(("results/clustered/%s/class-count.csv"%csv_name))

print("The result files are in the %s"%csv_name)

time_elapsed = time.time() - start_time
print("--- %s seconds ---" % (time_elapsed))
