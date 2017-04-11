# Lia Ristiana
# first step is to stratify the dataset into smaller dataset
# then we write it to csv file

import csv
import os

import pandas as pd
import numpy as np

import time
start_time = time.time()


input_file="/home/lia/Documents/FINAL-GEMASTIK/training.csv"

df = pd.read_csv(input_file)

df['class'] = df['class'].astype('category')

pd.set_option('float_format', '{:f}'.format)

# print(df.head(10))
# # find out the class frequency
# classCount = df['class'].value_counts(sort=False)
# print("class frequency\n%s."%classCount)
# print("total data = %s"%classCount.sum())

# DEFINE CLASS NAMES AS NEEDED
classNames = [1,2,3,4,5]

df_stratified = pd.DataFrame()

fraction = 0.01
for name in classNames:
	temp = df.loc[df['class']==name].sample(frac=fraction)
	df_stratified = pd.concat([df_stratified,temp])

print("RESULT ARRAY")
print(df_stratified.head(5))

# find out the stratified sampling data frequency
classCount = df_stratified['class'].value_counts(sort=False)
print("class frequency\n%s."%classCount)
print("total data = %s"%classCount.sum())

# LET'S COMPARE IT, BABY!
# print("STATS OF ORIGINAL DATASET")
# print(df.describe())
#
# print("STATS OF THE STRATIFIED-SAMPLED DATASET")
# print(df_stratified.describe())

# WRITE STRATIFIED DATA TO A CSV FILE

# csv_name = time.strftime("%Y-%m-%d %H:%M:%S")

dir_name = fraction
local_dir = "results/sampled/"
os.makedirs("results/sampled/%s"%(fraction), exist_ok=True)
df_stratified.to_csv(("results/sampled/%s/dataset.csv"%dir_name), sep=",", encoding="utf-8", index=False)
df_stratified.describe().to_csv(("results/sampled/%s/stats.csv"%dir_name), sep=",", encoding="utf-8")
classCount.to_csv(("results/sampled/%s/class-count.csv"%dir_name))
print("The result files are in the %s"%dir_name)

time_elapsed = time.time() - start_time
print("--- %s seconds ---" % (time_elapsed))
