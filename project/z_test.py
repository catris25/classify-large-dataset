# Lia Ristiana
# z test is a stastical method used to compare means difference

import math
import numpy as np
import pandas as pd

def calculate(df_population, df_sample):
    count_p = df_population.iloc[0].values
    means_p = df_population.iloc[1].values
    std_p = df_population.iloc[2].values

    count_s = df_sample.iloc[0].values
    means_s = df_sample.iloc[1].values
    std_s = df_sample.iloc[2].values

    for i in range(1, len(count_p)):
        # means of population and sample
        means1 = means_p[i]
        means2 = means_s[i]

        d = 0

        # variance of population and sample
        std1 = std_p[i]
        std2 = std_s[i]

        var1 = math.pow(std1,2)
        var2 = math.pow(std2,2)

        # number of population and sample
        n1 = count_p[i]
        n2 = count_s[i]

        standard_error = math.sqrt((var1/n1)+(var2/n2))
        z_value = ((means1-means2)-d)/standard_error
        print("attr%s: %s"%(i,z_value))

def main():
    input_file = "output/population_statistics.csv"
    df_population = pd.read_csv(input_file)

    input_file = "output/training_sampled/2017-05-07 23:12:26/stats.csv"
    df_sample = pd.read_csv(input_file)

    calculate(df_population, df_sample)

if __name__ == "__main__":
    main()
