# Lia Ristiana
# logistic regression with sklearn

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

import time

def classify(df_training, df_testing):
    # x = df_training[["attr1", "attr2", "attr3", "attr4", "attr5", "attr6", "attr7"]]
    # y = df_training["class"]

    training_attr = df_training.ix[:,[0,1,2,3,4,5,6]]
    training_target = df_training.ix[:,7]

    testing_attr = df_testing.ix[:,[0,1,2,3,4,5,6]]
    testing_target = df_testing.ix[:,7]

    model = LogisticRegression()
    model.fit(training_attr, training_target)

    prediction = model.predict(testing_attr)

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


def classify_all(training_dir, testing_dir, k_size):
    k = k_size

    total_accuracy = 0
    all_true_pos = 0
    all_sum_matrix = 0
    for i in range(k):

        file_name = i

        print("Cluster %s"%file_name)

        input_file="%s/%s.csv"%(training_dir,file_name)
        df_training = pd.read_csv(input_file)

        input_file="%s/%s.csv"%(testing_dir,file_name)
        df_testing = pd.read_csv(input_file)

        accu, true_pos, sum_matrix = classify(df_training, df_testing)
        all_true_pos += true_pos
        all_sum_matrix += sum_matrix

        print("------------------------")

    print("TP:",all_true_pos," all:",all_sum_matrix)
    print("P :",all_true_pos/all_sum_matrix)

    return all_true_pos, all_sum_matrix


def main():
    training_dir = "output/training_sampled/2017-05-07 23:12:26/training_set.csv"
    testing_dir = "output/testing_sampled/2017-05-07 23:12:32/testing-set.csv"
    k_size = 1

    df_training = pd.read_csv(training_dir)
    df_testing = pd.read_csv(testing_dir)
    print(classify(df_training, df_testing))


if __name__ == "__main__":
    main()

# the end
