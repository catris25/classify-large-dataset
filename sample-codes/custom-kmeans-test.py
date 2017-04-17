# Lia Ristiana
# k means clsustering custom code
# second step is to cluster the stratified dataset
# then we write it to csv file

import matplotlib.pyplot as plt
import numpy as np

# X = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11]])
# X = np.array([[1,2,5],[3,4,1],[5,4,4],[2,2,2],[7,8,2],[6,0,0],[9,9,2]])
X = np.array([[10.979,7.748],
            [9.352,8.743],
            [9.621,7.813],
            [9.666,12.14],
            [25.899,11.1],
            [24.012,8.39],
            [25.08,9.437],
            [29.752,10.36]])

# X = np.array([[3,2,5],[9,4,1],[0,4,4],[5,2,2],[1,8,2],[6,0,0]])
plt.scatter(X[:,0], X[:,1], s=150)
plt.show()

colors = 10*["g", "r", "c", "b", "k"]

class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
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
                sum_distances += min(distances)
                print(featureset,">>",distances,"\n")
                self.clustering[cluster_label].append(featureset)

            # print("sse: ",sum_distances)

            prev_centroids = dict(self.centroids)

            for cluster_label in self.clustering:
                self.centroids[cluster_label] = np.average(self.clustering[cluster_label], axis=0)
            # print("centroids: ",self.centroids)
            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid)/original_centroid*100.0)>self.tol:
                    optimized = False

            if optimized:
                break

        # def sse(self,data):



clr = K_Means()
clr.fit(X)

for centroid in clr.centroids:
    plt.scatter(clr.centroids[centroid][0], clr.centroids[centroid][1], marker="o", color="k", s=100)

for cluster in clr.clustering:
    color = colors[cluster]

    for featureset in clr.clustering[cluster]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=100)
plt.show()
for i in range(len(clr.clustering)):
    print(i,". ",clr.clustering[i])
print("centroid: ",clr.centroids)
