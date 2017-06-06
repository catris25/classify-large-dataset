# Lia Ristiana
# Support Vector Machine
import __main__

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn import svm
from sklearn.metrics import accuracy_score

import time

def classify(df_training, df_testing):
    training_attr = df_training.ix[:,[0,1,2,3,4,5,6]]
    training_target = df_training.ix[:,7]

    testing_attr = df_testing.ix[:,[0,1,2,3,4,5,6]]
    testing_target = df_testing.ix[:,7]

    model = svm.SVC()
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


def main():
    # training_dir = "output/training_sampled/2017-05-07 23:12:26/training_set.csv"
    # testing_dir = "output/testing_sampled/2017-05-07 23:12:32/testing-set.csv"

    # df_training = pd.read_csv(training_dir)
    # df_testing = pd.read_csv(testing_dir)

    data_dir = "/home/lia/Documents/work-it-out/weka data/training_set_ab.csv"

    df = pd.read_csv(data_dir)
    df.drop(['original'],1,inplace=True)
    df_training = df.sample(frac=0.66, random_state=1)
    df_testing = df.loc[~df.index.isin(df_training.index)]

    print(classify(df_training, df_testing))

if __name__ == "__main__":
    main()
