# Lia Ristiana
# k means clsustering custom code
# second step is to cluster the stratified dataset
# then we write it to csv file

import matplotlib.pyplot as plt
import numpy as np

X = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11]])
# X = np.array([[3,1],[1,1],[9,3],[5,2],[8,8],[5,6],[3,3],[7,4],[5,5],[6,7],[1,4],[5,0]])
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
            self.classifications= {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                # print(featureset,"==",distances)
                classification = distances.index(min(distances))
                # print(classification)
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid)/original_centroid*100.0)>self.tol:
                    optimized = False

            if optimized:
                break

    def predict(self,data):
        distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
        classifications = distances.index(min(distances))
        return classification

clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker="o", color="k", s=150)

for classification in clf.classifications:
    color = colors[classification]

    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150)

plt.show()
