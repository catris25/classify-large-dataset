# Lia Ristiana
# first step is to stratify the dataset into

import pandas as pd
import numpy as np

import time
start_time = time.time()


input_file="/home/lia/Documents/FINAL-GEMASTIK/training.csv"

df = pd.read_csv(input_file)

df['class'] = df['class'].astype('category')

pd.set_option('float_format', '{:f}'.format)

print(df.head(10))
# find out the class frequency
classCount = df['class'].value_counts(sort=False)
print("class frequency\n%s."%classCount)
print("sum = %s"%classCount.sum())

# DEFINE CLASS NAMES AS NEEDED
classNames = [1,2,3,4,5]

df_stratified = pd.DataFrame()

for name in classNames:
	temp = df.loc[df['class']==name].sample(frac=0.0005)
	df_stratified = pd.concat([df_stratified,temp])

print("RESULT ARRAY")
print(df_stratified.head(5))

# find out the stratified sampling data frequency
classCount = df_stratified['class'].value_counts(sort=False)
print("class frequency\n%s."%classCount)
print("sum = %s"%classCount.sum())

# LET'S COMPARE IT, BABY!
print("STATS OF ORIGINAL DATASET")
print(df.describe())

print("STATS OF THE STRATIFIED-SAMPLED DATASET")
print(df_stratified.describe())

time_elapsed = time.time() - start_time
print("--- %s seconds ---" % (time_elapsed))
