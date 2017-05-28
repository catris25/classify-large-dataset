# Lia Ristiana
# feature subset selection
# finding out which combination of features produce the best accuracy
# using naive bayes classifier

from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

import pandas as pd

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
    training_attr = training_set.ix[:,[0,1,2,3,4,5,6]]
    training_target = training_set.ix[:,7]

    testing_set = df.loc[~df.index.isin(training_set.index)]
    testing_attr = testing_set.ix[:,[0,1,2,3,4,5,6]]


    testing_target = testing_set.ix[:,7]

    for i in range(1):

        # df_attr = df.ix[i]
        training_attr = training_attr.ix[:,[1,2,4,5]]
        testing_attr = testing_attr.ix[:,[1,2,4,5]]
        # print(testing_attr.head())
        # return
        # testing_attr = testing_attr.ix[i]
        result = naive_bayes(training_attr,training_target, testing_attr, testing_target)

        print("attr-%s"%i)
        print(result)
        print("----------------")




def main():


    input_file="output/training_sampled/2017-05-07 23:12:26/training_set.csv"

    df = pd.read_csv(input_file)
    brute_force(df)


if __name__ == "__main__":
    main()

# the end