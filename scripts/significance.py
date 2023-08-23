# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv

# %%
# Import csv as panda dataframe

import pandas as pd
df = pd.read_csv("/Users/nadine/repos/statistics/data/size-Pd_June2023.csv") 
print(df)        

# %%
# Replace NaN with empty string

#df = df.fillna('')

# %%
# Transform colum to list

group1 = df.iloc[:, 4].tolist()

group2 = df.iloc[:, 5].tolist()


# %%
# unpaired t-test

from scipy.stats import ttest_ind, ttest_ind_from_stats
from scipy.special import stdtr


output = ttest_ind(group1, group2, nan_policy= "omit") #ttest_ind = from independent groups; 'omit' ignores NaN


# %% test

a = [1,4,2,7,4,4,5,2]
b = [6,6,2,4,5,5,1,1]
output = ttest_ind(a, b) 


# %%
