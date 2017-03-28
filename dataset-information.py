import numpy
import pandas
import matplotlib.pyplot as plt 
import random

import time
start_time = time.time()

input_file="/home/lia/Documents/FINAL-GEMASTIK/training.csv"

df = pandas.read_csv(input_file)

print("DATA SHAPE")
print(df.shape)

df['class'] = df['class'].astype('category')
print("COLUMN TYPES")
print(df.dtypes)

pandas.set_option('float_format', '{:f}'.format)

# DATA STATS
# print("DATA STATS")
# print(df.describe())


# dataStats = df['attr1'].describe().
# print(dataStats)

# print("CATEGORY STATS")
# classCount = df['class'].value_counts(sort=False)

# # accessing series
# print(classCount.values)
# print(classCount.index.tolist())

# plt.bar(classCount.index.tolist(),classCount.values)
# plt.xlabel('class')
# plt.ylabel('freq')
# plt.title('Class frequency')
# plt.show()


time_elapsed = time.time() - start_time
print("--- %s seconds ---" % (time_elapsed))
