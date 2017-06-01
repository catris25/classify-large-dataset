# Lia Ristiana
# feature subset selection
# finding out which combination of features produce the best accuracy
# using naive bayes classifier

from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

import itertools
import pandas as pd
import numpy as np

def naive_bayes(training_attr,training_target, testing_attr, testing_target):

    model = GaussianNB()
    model.fit(training_attr, training_target)

    prediction = model.predict(testing_attr)

    # print("confusion matrix")
    cm = metrics.confusion_matrix(testing_target, prediction)
    # print(cm)

    accu = accuracy_score(prediction, testing_target)
    # print("accuracy")
    # print(accu)
    return accu

def brute_force(df):
    # df_attr = df.ix[:,[0,1,2,3,4,5,6]]
    # df_target = df.ix[:,7]
    # attr_names = ['attr1', 'attr2', 'attr3', 'attr4','attr5', 'attr6', 'attr7']

    # print(df_attr['attr3'].head())
    training_set = df.sample(frac=0.66, random_state=1)
    temp_training_attr = training_set.ix[:,[0,1,2,3,4,5,6,7,8]]
    training_target = training_set.ix[:,9]

    testing_set = df.loc[~df.index.isin(training_set.index)]
    temp_testing_attr = testing_set.ix[:,[0,1,2,3,4,5,6,7,8]]


    testing_target = testing_set.ix[:,9]



    # for i in range(1,7):
    #     print(list(itertools.combinations('1234567',i)))

    attrs = [0,1,2,3,4,5,6,7,8]
    # attrs = ['attr1', 'attr2', 'attr3', 'attr4', 'attr5', 'attr6', 'attr7']
    j =1

    accu = []
    for i in range(1,len(attrs)+1):
        for c in itertools.combinations(attrs,i):
            print(j,"",c)

            training_attr = temp_training_attr.ix[:,c]
            testing_attr = temp_testing_attr.ix[:,c]
            # print(training_attr.head())
            j+=1

            result = naive_bayes(training_attr,training_target, testing_attr, testing_target)
            print(result)

            accu.append(result)

    print("maximum ",max(accu))

def main():


    # input_file="output/training_sampled/2017-05-07 23:12:26/training_set.csv"
    input_file = "/home/lia/Documents/work-it-out/weka data/training_set_ndvi.csv"

    df = pd.read_csv(input_file)
    brute_force(df)


if __name__ == "__main__":
    main()

# the end
