# Lia Ristiana
# Naive Bayes classification implementation using sklearn
# training set is obtained from the clustered data
# testing set is obtained from the sampled data

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

import time
start_time = time.time()

dir_name = "2017-04-03 01:06:26"
file_name = "2"
input_file="results/training/%s/%s.csv"%(dir_name,file_name)
df_training = pd.read_csv(input_file)

dir_name = "2017-04-03 01:06:26"
file_name = "2"
input_file="results/testing/%s/%s.csv"%(dir_name,file_name)
df_testing = pd.read_csv(input_file)

model = GaussianNB()
model.fit(training_attr, training_target)

prediction = model.predict(testing_attr)

print("Cluster %s"%file_name)

print("confusion matrix")
print(metrics.confusion_matrix(testing_target, prediction))

accu = accuracy_score(prediction, testing_target)
print("accuracy")
print(accu)

time_elapsed = time.time() - start_time
print("--- %s seconds ---" % (time_elapsed))
