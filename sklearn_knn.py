# Lia Ristiana
# KNN classification implementation using sklearn

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import time
start_time = time.time()

training_dir_name = "results/sampled/2017-05-04 23:23:02/dataset.csv"
testing_dir_name = "results/testing/2017-05-05 01:44:45/testing-set.csv"

def classify_knn(df_training, df_testing,n_size):
    training_attr = df_training.ix[:,[0,1,2,3,4,5,6]]
    training_target = df_training.ix[:,7]

    testing_attr = df_testing.ix[:,[0,1,2,3,4,5,6]]
    testing_target = df_testing.ix[:,7]

    model = KNeighborsClassifier(n_neighbors=n_size)
    model.fit(training_attr, training_target)

    prediction = model.predict(testing_attr)

    print("confusion matrix")
    cm = metrics.confusion_matrix(testing_target, prediction)
    print(cm)

    # get true positive
    true_pos = np.diag(cm).sum()
    sum_matrix = cm.sum()

    print("TP/all: %s/%s"%(true_pos,sum_matrix))

    accu = accuracy_score(prediction, testing_target)
    print("accuracy:",accu)
    # return accu, true_pos, sum_matrix

df_training = pd.read_csv(training_dir_name)
df_testing = pd.read_csv(testing_dir_name)
n_size = 5
classify_knn(df_training, df_testing,n_size)

time_elapsed = time.time() - start_time
print("--- %s seconds ---" % (time_elapsed))
