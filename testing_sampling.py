# Lia Ristiana
# next step is to stratify the dataset for sampling
# with fixed number of data from each class
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

# DEFINE CLASS NAMES AS NEEDED
classNames = [1,2,3,4,5]

df_stratified = pd.DataFrame()

for name in classNames:
	temp = df.loc[df['class']==name].sample(n=100)
	df_stratified = pd.concat([df_stratified,temp])

print("RESULT ARRAY")
print(df_stratified.head(5))
classCount = df_stratified['class'].value_counts(sort=False)
print("class frequency\n%s."%classCount)
print("total data = %s"%classCount.sum())

csv_name = time.strftime("%Y-%m-%d %H:%M:%S")
local_dir = "results/testing/"
os.makedirs("%s%s"%(local_dir,csv_name), exist_ok=True)
df_stratified.to_csv(("%s%s/testing-set.csv"%(local_dir,csv_name)), sep=",", encoding="utf-8", index=False)
df_stratified.describe().to_csv(("%s%s/stats.csv"%(local_dir,csv_name)), sep=",", encoding="utf-8")

print("The result files are in the %s"%csv_name)
