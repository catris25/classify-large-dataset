# Lia Ristiana
# Naive Bayes classification implementation using sklearn
# training set is obtained from the clustered data
# testing set is obtained from the sampled data

import __main__

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

import time
start_time = time.time()

training_dir_name = __main__.main_training_dir
testing_dir_name = __main__.main_testing_dir

k= __main__.main_k

def classify_nb(df_training, df_testing):
    training_attr = df_training.ix[:,[0,1,2,3,4,5,6]]
    training_target = df_training.ix[:,7]

    testing_attr = df_testing.ix[:,[0,1,2,3,4,5,6]]
    testing_target = df_testing.ix[:,7]

    model = GaussianNB()
    model.fit(training_attr, training_target)

    prediction = model.predict(testing_attr)

    print("Cluster %s"%file_name)

    print("confusion matrix")
    cm = metrics.confusion_matrix(testing_target, prediction)
    print(cm)

    accu = accuracy_score(prediction, testing_target)
    print("accuracy")
    print(accu)

    # get true positive
    true_pos = np.diag(cm).sum()
    sum_matrix = cm.sum()

    print("TP/all: %s/%s"%(true_pos,sum_matrix))

    return accu, true_pos, sum_matrix

total_accuracy = 0
all_true_pos = 0
all_sum_matrix = 0
for i in range(k):
    file_name = i
    input_file="%s/%s.csv"%(training_dir_name,file_name)
    df_training = pd.read_csv(input_file)

    input_file="%s/%s.csv"%(testing_dir_name,file_name)
    df_testing = pd.read_csv(input_file)

    accu, true_pos, sum_matrix = classify_nb(df_training, df_testing)
    all_true_pos += true_pos
    all_sum_matrix += sum_matrix

    print("------------------------")

print("TP:",all_true_pos," all:",all_sum_matrix)
print("P :",all_true_pos/all_sum_matrix)

time_elapsed = time.time() - start_time
print("--- %s seconds ---" % (time_elapsed))
