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

df.boxplot(vert=False)
# %%
