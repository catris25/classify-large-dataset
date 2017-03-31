import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
import sklearn.metrics as sm

import pandas as pd
import numpy as np

import time
start_time = time.time()

input_file="datasets/iris.csv"
df = pd.read_csv(input_file)

x = df.ix[:,[0,1,2,3]]
y = df.ix[:,4]

model = KMeans(n_clusters=5)
model.fit(x)

centroids = model.cluster_centers_
labels = model.labels_

colors = ["g.", "r.", "c.", "b.", "k."]

# for i in range(len(x)):
# 	plt.plot(x[i][0], x[i][1], colors[labels[i]], markersize=25)
plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=150, linewidth=5)
plt.show()
