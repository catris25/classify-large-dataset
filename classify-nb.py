# Lia Ristiana
# Naive Bayes classification implementation using sklearn

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

import time
start_time = time.time()

input_file="results/clustered/2017-04-02 21:07:24/1.csv"
df = pd.read_csv(input_file)

# split data into training and testing set

# training
training_set = df.sample(frac=0.66, random_state=1)
training_attr = training_set.ix[:,[0,1,2,3,4,5,6]]
training_target = training_set.ix[:,7]

# testing
testing_set = df.loc[~df.index.isin(training_set.index)]
testing_attr = testing_set.ix[:,[0,1,2,3,4,5,6]]
testing_target = testing_set.ix[:,7]

model = GaussianNB()
model.fit(training_attr, training_target)

prediction = model.predict(testing_attr)

print("confusion matrix")
print(metrics.confusion_matrix(testing_target, prediction))

accu = accuracy_score(prediction, testing_target)
print("accuracy")
print(accu)

time_elapsed = time.time() - start_time
print("--- %s seconds ---" % (time_elapsed))
