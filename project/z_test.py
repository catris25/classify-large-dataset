# Lia Ristiana
# z test is a stastical method used to compare means difference

import math
import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.special as ss

def calculate(df_population, df_sample):
    count_p = df_population.iloc[0].values
    means_p = df_population.iloc[1].values
    std_p = df_population.iloc[2].values

    count_s = df_sample.iloc[0].values
    means_s = df_sample.iloc[1].values
    std_s = df_sample.iloc[2].values

    # calculate the z value for each iteration (attribute)
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
        p_val = st.norm.sf(abs(z_value))*2

        # px = 1 - ss.ndtr(z_value)
        # pval = st.t.sf(np.abs(z_value),n2-1)*2

        # as it turns out the calculation of pval and p_val are the same
        # and px is one tail

        # print("attr%s: %s, probability: %s, %s, %s"%(i,z_value,p_val,px, pval))
        print("attr-%s z = %s, p = %s"%(i,z_value, p_val))

        # calculate the degrees of freedom
        # pembilang = (var1/n1+var2/n2)**2
        # penyebut1 = ((var1/n1)**2)/(n1-1)
        # penyebut2 = ((var2/n2)**2)/(n2-1)
        # penyebut = penyebut1+penyebut2
        # d_freedom = pembilang/penyebut
        # print("degrees of freedom",d_freedom)

def main():
    input_file = "output/population_statistics.csv"
    df_population = pd.read_csv(input_file)

    input_file = "output/training_sampled/2017-05-28 22:50:21/training_set.csv"
    df_sample = pd.read_csv(input_file)

    print("input file:",input_file)
    calculate(df_population, df_sample)

    # print("test",st.norm.cdf(1.64))
    # print(st.norm.sf(abs(1.64))*2)
    # print("test",st.t.ppf(0.93339648357,999))

if __name__ == "__main__":
    main()
