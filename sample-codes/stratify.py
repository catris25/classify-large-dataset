# Lia Ristiana
# learning to do stratify random sampling on a dataset

import pandas as pd
import numpy as np

input_file="../datasets/iris.csv"

df = pd.read_csv(input_file)

# find out the class frequency
classCount = df['class'].value_counts(sort=True)
print("class frequency\n%s."%classCount)

# DEFINE CLASS NAMES AS NEEDED
classNames = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

df_stratified = pd.DataFrame()

for name in classNames:
	temp = df.loc[df['class']==name].sample(frac=0.1)
	df_stratified = pd.concat([df_stratified,temp])
	
print("RESULT ARRAY")
print(df_stratified.head(5))

# find out the stratified sampling data frequency
classCount = df_stratified['class'].value_counts()
print("class frequency\n%s."%classCount)
