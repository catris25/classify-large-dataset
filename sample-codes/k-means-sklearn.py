from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import pylab as pl
import pandas as pd

input_file="/home/lia/Documents/FINAL-GEMASTIK/training.csv"
df = pd.read_csv(input_file)

X = df.ix[:,[0,1,2,3,4,5,6]]
y = df.ix[:,7]

pca = PCA(n_components=2).fit(df)

pca_2d = pca.transform(df)
pl.figure('Reference Plot')
pl.scatter(pca_2d[:, 0], pca_2d[:, 1], c=y)
kmeans = KMeans(n_clusters=3, random_state=111)
kmeans.fit(X)
pl.figure('K-means with 3 clusters')
pl.scatter(pca_2d[:, 0], pca_2d[:, 1], c=kmeans.labels_)
pl.show()