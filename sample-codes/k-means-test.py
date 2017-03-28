import pandas
import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

import time
start_time = time.time()

input_file="/home/lia/Documents/FINAL-GEMASTIK/training.csv"
df = pandas.read_csv(input_file)

# print(df.head())

X = df.ix[:,7]

# plt.scatter(X[:,0], X[:,1], s=150)
# plt.show()

clf = KMeans(n_clusters=4)
clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_

colors = ["g.", "r.", "c.", "b.", "k."]

for i in range(len(X)):
	plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=25)
plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=150, linewidth=5)
plt.show()