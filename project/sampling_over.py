# Lia Ristiana
# next step is to stratify the dataset for sampling
# with fixed number of data from each class
# then we write it to csv file

import csv
import os

import pandas as pd
import numpy as np

import time

def sampling(input_file, n_size):
    df = pd.read_csv(input_file)

    df['class'] = df['class'].astype('category')

    pd.set_option('float_format', '{:f}'.format)

    # DEFINE CLASS NAMES AS NEEDED
    classNames = [1,2,3,4,5]

    df_stratified = pd.DataFrame()

    ori_s = [1130,3539,1130,5380,4079]
    over_s = [1130,3539,292,5380,4079]

    i=0
    for name in classNames:
        n_size = ori_s[i] - over_s[i]
    	temp = df.loc[df['class']==name].sample(n=n_size)
    	df_stratified = pd.concat([df_stratified,temp])

        i=+1

    print("RESULT ARRAY")
    print(df_stratified.head(5))
    classCount = df_stratified['class'].value_counts(sort=False)
    print("class frequency\n%s."%classCount)
    print("total data = %s"%classCount.sum())

    # csv_name = time.strftime("%Y-%m-%d %H:%M:%S")
    # local_dir = "output/testing_sampled/"
    # os.makedirs("%s%s"%(local_dir,csv_name), exist_ok=True)
    # df_stratified.to_csv(("%s%s/testing-set.csv"%(local_dir,csv_name)), sep=",", encoding="utf-8", index=False)
    # df_stratified.describe().to_csv(("%s%s/stats.csv"%(local_dir,csv_name)), sep=",", encoding="utf-8")
    #
    # print("The result files are in the %s"%csv_name)
    #
    # testing_sampled_dir = local_dir+csv_name+"/testing-set.csv"
    # return testing_sampled_dir

def main():
    # input_file="output/testing_sampled/2017-06-06 22:57:30/testing-set.csv"
    # input_file = 
    n_size = 10000
    sampl = sampling(input_file, n_size)
    print("result:",sampl)

if __name__ == "__main__":
    main()

# the end
