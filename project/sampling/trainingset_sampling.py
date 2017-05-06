# Lia Ristiana
# first step is to stratify the dataset into smaller dataset
# then we write it to csv file

import csv
import os

import pandas as pd
import numpy as np
import time

def sampling(input_file):

    df = pd.read_csv(input_file)
    df['class'] = df['class'].astype('category')

    pd.set_option('float_format', '{:f}'.format)

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

    # WRITE STRATIFIED DATA TO A CSV FILE

    dir_name = time.strftime("%Y-%m-%d %H:%M:%S")
    local_dir = "../output/sampled"
    os.makedirs("../output/sampled/%s"%dir_name, exist_ok=True)
    df_stratified.to_csv(("../output/sampled/%s/dataset.csv"%dir_name), sep=",", encoding="utf-8", index=False)
    df_stratified.describe().to_csv(("../output/sampled/%s/stats.csv"%dir_name), sep=",", encoding="utf-8")
    classCount.to_csv(("../output/sampled/%s/class-count.csv"%dir_name))

    sampled_training_dir = "%s/%s"%(local_dir,dir_name)

    return sampled_training_dir

def main():
    input_file="/home/lia/Documents/FINAL-GEMASTIK/training.csv"
    print(sampling(input_file))

if __name__ == "__main__":
    main()

# the end
