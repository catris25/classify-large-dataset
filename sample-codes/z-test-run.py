# Lia Ristiana
import matplotlib.pyplot as plt
from scipy.spatial import distance

import numpy as np
import pandas as pd
import math

import time
start_time = time.time()

x1 = np.array([[10.979,7.748],
            [9.352,8.743],
            [9.621,7.813],
            [9.666,12.14],
            [25.899,11.1],
            [24.012,8.39],
            [25.08,9.437],
            [29.752,10.36]])

x2 = np.array([[9.666,12.14],[25.899,11.1],[24.012,8.39]])

standard_error = np.std(x2)/math.sqrt(len(x1))
print(standard_error)
z_value = (np.mean(x2)-np.mean(x1))/standard_error
print(z_value)
