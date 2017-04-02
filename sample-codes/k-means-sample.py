import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

X = np.array([[1,2],
			[1.5,1.8],
			[5,8],
			[8,7],
			[2,11],
			[6,2],
			[1,8],
			[7,3],
			[9,6],
			[1,5],
			[10,0],
			[11,1],
			[6,5]])

plt.scatter(X[:,0], X[:,1], s=150)
plt.show()

clf = KMeans(n_clusters=5)
clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_

colors = ["g.", "r.", "c.", "b.", "k."]

for i in range(len(X)):
	plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=25)
plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=150, linewidth=5)
plt.show()
