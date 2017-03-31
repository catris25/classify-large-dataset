import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
import sklearn.metrics as sm

import pandas as pd
import numpy as np

import time
start_time = time.time()

# input_file="datasets/iris.csv"
# df = pandas.read_csv(input_file)

iris = datasets.load_iris()

x = pd.DataFrame(iris.data)
x.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']

y = pd.DataFrame(iris.target)
y.columns = ['Targets']

# K Means Clutering
model = KMeans(n_clusters=3)
model.fit(x)

print(model.labels_)

# View the results
# Set the size of the plot
plt.figure(figsize=(8,4))

# Create a colormap
colormap = np.array(['red', 'lime', 'black'])

# Plot the Original Classifications
plt.subplot= (1, 2, 1)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[y.Targets], s=40)
plt.title('Real Classification')

# plt.show()

# Plot the Models Classifications
plt.subplot=(1, 2, 2)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[model.labels_], s=40)
plt.title('K Mean Classification')

# plt.show()

# The fix, we convert all the 1s to 0s and 0s to 1s.
predY = np.choose(model.labels_, [1, 0, 2]).astype(np.int64)
print (model.labels_)
print (predY)

print("accuracy:",sm.accuracy_score(y, predY))
print("confusion matrix")
print(sm.confusion_matrix(y, predY))
