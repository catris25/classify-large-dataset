import pandas as pd
import numpy as np

# s = pd.Series(np.random.randn(50))
# df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))

input_file="../datasets/iris.csv"

df = pd.read_csv(input_file)

classCount = df['class'].value_counts(sort=True)
print("class frequency\n%s."%classCount)

classNames = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# df_stratify = df.loc[df['class'].isin(classNames)].sample(frac=0.1)

df_stratify = list()

for name in classNames:
	df_stratify.extend(df.loc[df['class']==name].sample(frac=0.1))
	# print(df.loc[df['class']==name].sample(frac=0.1))

print(df.head(10))
print(df_stratify)

classCount = df_stratify['class'].value_counts()
print("class frequency\n%s."%classCount)



