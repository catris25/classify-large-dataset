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

def classify(df_training, df_testing, n_size):
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
    return accu, true_pos, sum_matrix


def classify_all(training_dir, testing_dir, k_size, n_size):
    k_cluster = k_size

    total_accuracy = 0
    all_true_pos = 0
    all_sum_matrix = 0

    for i in range(k_cluster):
        file_name = i

        print("Cluster %s"%file_name)

        input_file="%s/%s.csv"%(training_dir,file_name)
        df_training = pd.read_csv(input_file)

        input_file="%s/%s.csv"%(testing_dir,file_name)
        df_testing = pd.read_csv(input_file)

        accu, true_pos, sum_matrix = classify(df_training, df_testing, n_size)
        all_true_pos += true_pos
        all_sum_matrix += sum_matrix

        print("------------------------")

    print("TP:",all_true_pos," all:",all_sum_matrix)
    print("P :",all_true_pos/all_sum_matrix)

    return all_true_pos, all_sum_matrix

def main():
    # training_dir = "output/training_clustered/3clusters/"
    # testing_dir = "output/testing_clustered/3clusters/"
    k_size = 1
    n_size = 7

    training_dir = "output/training_sampled/2017-05-07 23:12:26/training_set.csv"
    testing_dir = "output/testing_sampled/2017-05-07 23:12:32/testing-set.csv"

    df_training = pd.read_csv(training_dir)
    df_testing = pd.read_csv(testing_dir)
    print(classify(df_training, df_testing, n_size))

    # clsr = classify_all(training_dir,testing_dir, k_size, n_size)

if __name__ == "__main__":
    main()


time_elapsed = time.time() - start_time
print("--- %s seconds ---" % (time_elapsed))
