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

    # find out the stratified sampling data frequency
    classCount = df_stratified['class'].value_counts(sort=False)
    print("class frequency\n%s."%classCount)
    print("total data = %s"%classCount.sum())

    # WRITE STRATIFIED DATA TO A CSV FILE

    dir_name = time.strftime("%Y-%m-%d %H:%M:%S")
    local_dir = "output/training_sampled"

    os.makedirs("%s/%s"%(local_dir, dir_name), exist_ok=True)
    df_stratified.to_csv(("%s/%s/training_set.csv"%(local_dir, dir_name)), sep=",", encoding="utf-8", index=False)
    df_stratified.describe().to_csv(("%s/%s/stats.csv"%(local_dir,dir_name)), sep=",", encoding="utf-8")
    classCount.to_csv(("%s/%s/class-count.csv"%(local_dir,dir_name)))

    training_sampled_dir = "%s/%s/training_set.csv"%(local_dir,dir_name)

    return training_sampled_dir

def main():
    input_file="/home/lia/Documents/FINAL-GEMASTIK/training.csv"
    print(sampling(input_file))

if __name__ == "__main__":
    main()

# the end
