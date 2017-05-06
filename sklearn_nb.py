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

training_dir_name = "results/sampled/2017-04-13 12:33:41/dataset.csv"
testing_dir_name = "results/testing/2017-05-05 01:44:45/testing-set.csv"

def classify_nb(df_training, df_testing):
    training_attr = df_training.ix[:,[0,1,2,3,4,5,6]]
    training_target = df_training.ix[:,7]

    testing_attr = df_testing.ix[:,[0,1,2,3,4,5,6]]
    testing_target = df_testing.ix[:,7]

    model = GaussianNB()
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
classify_nb(df_training, df_testing)

time_elapsed = time.time() - start_time
print("--- %s seconds ---" % (time_elapsed))
