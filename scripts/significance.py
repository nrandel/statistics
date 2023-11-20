# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv

# %%
# Import csv as panda dataframe

import pandas as pd
df = pd.read_csv("/Users/nadine/Documents/paper/Naomi-NS-maturation/BF_EdU-larvae_size-merge_um.csv") 
print(df)        

# %%

# Transform colum to list

group1 = df.iloc[:, 17].tolist()

group2 = df.iloc[:, 18].tolist()


# %%
# unpaired t-test

from scipy.stats import ttest_ind, ttest_ind_from_stats
from scipy.special import stdtr


output = ttest_ind(group1, group2, nan_policy= "omit") #ttest_ind = from independent groups; 'omit' ignores NaN

print(output)

# %%
