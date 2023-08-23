# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv

# %%
# Import csv as panda dataframe

import pandas as pd
df = pd.read_csv("/Users/nadine/repos/statistics/data/size-Pd_June2023-um.csv") 
print(df) 

# Values in current dataframe are in some value from Fiji!!

# %%
# box and wisker plot with label on left side

df.boxplot(vert=False, grid = False, color=dict(boxes='k', whiskers='k', medians='b', caps='k'))

# %%
