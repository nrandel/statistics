# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv

# %%
# Import csv as panda dataframe

import pandas as pd
#df = pd.read_csv("/Users/nadine/repos/statistics/data/size-Pd_June2023-um.csv") 
df = pd.read_csv("/Users/nadine/Documents/paper/Naomi-NS-maturation/BF_EdU_merged_um-grouped.csv") 
print(df) 

# Values in current dataframe are in some value from Fiji!!

# %%
# box and wisker plot with label on left side
#from matplotlib.pyplot import figure
#figure(figsize=(10, 8), dpi=100)

ax = df.boxplot(vert=False, grid = False, color=dict(boxes='k', whiskers='k', medians='b', caps='k'), showfliers=False) #showfliers=False hide outlier datapoints

# %%
# Export plot as svg

ax.figure.savefig('/Users/nadine/Documents/paper/Naomi-NS-maturation/generated_plots/feeding_experiment.svg')
# %%
# diff figure style 
