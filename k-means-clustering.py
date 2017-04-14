# Lia Ristiana
# second step is to cluster the stratified dataset
# then we write it to csv file

import os

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

import time
start_time = time.time()

dir_name = "2017-04-13 12:33:41"
input_file="results/sampled/%s/dataset.csv"%dir_name

df = pd.read_csv(input_file)

x = df.ix[:,[0,1,2,3,4,5,6]]
y = df.ix[:,7]

k=10

model = KMeans(n_clusters=k)
model.fit(x)

centroids = model.cluster_centers_
labels = model.labels_

# INSERT THE LABEL OF CLUSTER TO DATAFRAME
label_col = pd.Series(labels)
x["class"] = y
x["cluster"] = label_col.values

# SAVE THE WHOLE DATA TO CSV
# dir_name = "k%s"%k
# os.makedirs("results/clustered/%s"%dir_name, exist_ok=True)
# x.to_csv(("results/clustered/%s/clustered-dataset.csv"%dir_name), sep=",", encoding="utf-8", index=False)

# SAVE THE DATA TO ITS OWN CSV BASED ON ITS CLUSTER
clusterNames = list(range(0,k))

dir_name = "%sclusters"%k
os.makedirs("results/clustered/%s"%dir_name, exist_ok=True)

# store the centroids
df_centroids = pd.DataFrame(centroids,columns=['attr1', 'attr2', 'attr3', 'attr4', 'attr5', 'attr6', 'attr7'])
df_centroids.to_csv("results/clustered/%s/centroids.csv"%dir_name, index=False)

for name in clusterNames:
    temp = x.loc[x['cluster']==name]
    temp.to_csv("results/clustered/%s/%s.csv"%(dir_name,name), index=False)
    classCount = temp['class'].value_counts(sort=False)
    print(classCount)
    # df_class = classCount.to_frame()
    # df_class.to_csv("results/clustered/%s/class-count-%s.csv"%(dir_name,name), sort=False)


print("The result files are in the %s"%(dir_name))

time_elapsed = time.time() - start_time
print("--- %s seconds ---" % (time_elapsed))
