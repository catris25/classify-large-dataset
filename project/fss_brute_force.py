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
    training_attr = training_set.ix[:,[0,1,2,3,4,5,6]]
    training_target = training_set.ix[:,7]

    testing_set = df.loc[~df.index.isin(training_set.index)]
    testing_attr = testing_set.ix[:,[0,1,2,3,4,5,6]]


    testing_target = testing_set.ix[:,7]

    

    # for i in range(1,7):
    #     print(list(itertools.combinations('1234567',i)))

    attrs = [0,1,2,3,4,5,6]
    # attrs = ['attr1', 'attr2', 'attr3', 'attr4', 'attr5', 'attr6', 'attr7']
    j =1
    for i in range(1,len(attrs)+1):
        for c in itertools.combinations(attrs,i):
            temp = np.array(c)
            print(j,"",c)
            print(temp[0])
            training_attr = training_attr.ix[:,temp[0]]
            print(training_attr.head())

            j+=1
            # training_attr = training_attr.ix[:,[c]]
            # print(training_attr.head())
            # training_attr = training_attr.ix[:,[subset]]
            # testing_attr = testing_attr.ix[:,[subset]]
            # result = naive_bayes(training_attr,training_target, testing_attr, testing_target)
            # print(result)



        # training_attr = training_attr.ix[:,[i]].as_matrix()
        # testing_attr = testing_attr.ix[:,[i]].as_matrix()


        # result = naive_bayes(training_attr,training_target, testing_attr, testing_target)
        # print(result)


    # for i in range(7):
    #
    #     # df_attr = df.ix[i]
    #     training_attr = training_attr.ix[:,[i]]
    #
    #     testing_attr = testing_attr.ix[:,[i]]
    #     # print(testing_attr.head(10))
    #     # return
    #     # print(testing_attr.head())
    #     # return
    #     # testing_attr = testing_attr.ix[i]
    #     result = naive_bayes(training_attr,training_target, testing_attr, testing_target)
    #
    #     print("attr-%s"%i)
    #     print(result)
    #     print("----------------")




def main():


    input_file="output/training_sampled/2017-05-07 23:12:26/training_set.csv"

    df = pd.read_csv(input_file)
    brute_force(df)


if __name__ == "__main__":
    main()

# the end
