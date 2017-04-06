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

dir_name = "2017-04-06 11:11:08"
file_name = "4"
input_file="results/clustered/%s/%s.csv"%(dir_name,file_name)
df_training = pd.read_csv(input_file)

dir_name = "2017-04-06 11:14:01"
file_name = "4"
input_file="results/testing-set/%s/%s.csv"%(dir_name,file_name)
df_testing = pd.read_csv(input_file)

training_attr = df_training.ix[:,[0,1,2,3,4,5,6]]
training_target = df_training.ix[:,7]

testing_attr = df_testing.ix[:,[0,1,2,3,4,5,6]]
testing_target = df_testing.ix[:,7]

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
