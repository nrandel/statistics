# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import csv

# %%
# Import csv as panda dataframe

import pandas as pd
df = pd.read_csv("/Users/nadine/Documents/paper/Naomi-NS-maturation/BF_EdU-larvae/BF_EdU-larvae_size-merge_um.csv") 
print(df)        

# %%
# Transform column to list (used for unpaired t-test below)

group1 = df.iloc[:, 17].tolist()
group2 = df.iloc[:, 18].tolist()

#%% Choose columnns by name
#column_names = ['3dpf cntr', '3dpf Tetraselmis' ,'3dpf G. marina' ,'4dpf cntr' ,'4dpf Tetraselmis' ,'4dpf G. marina' ,'5dpf cntr' ,'5dpf Tetraselmis' ,'5dpf G. marina' ,'6dpf cntr', '6dpf Tetraselmis', '6dpf G. marina', '7dpf cntr', '7dpf Tetraselmis', '7dpf G. marina' , '8dpf cntr' ,'8dpf Tetraselmis' ,'8dpf G. marina']  
column_names = ['3dpf cntr']
selected_columns = df[column_names]

# %%
# Test normal distribution with Shapiro-Wilk test. (If p>0.05, than data are normally distributed)

# Visual inspection
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
for col in selected_columns.columns:
    # Ignore NaN values for each column
    data = selected_columns[col].dropna()
    plt.hist(data, bins=30, density=True, alpha=0.6, label=col)
plt.title('Histogram of Data')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

plt.subplot(1, 2, 2)
for col in selected_columns.columns:
    # Ignore NaN values for each column
    data = selected_columns[col].dropna()
    stats.probplot(data, dist="norm", plot=plt)
plt.title('Q-Q plot')

plt.tight_layout()
plt.show()

# Statistical test (Shapiro-Wilk test for each selected column)
for col in selected_columns.columns:
    # Ignore NaN values for each column
    data = selected_columns[col].dropna()
    statistic, p_value = stats.shapiro(data)
    print(f"Shapiro-Wilk test for column '{col}':")
    print("Statistic:", statistic)
    print("p-value:", p_value)
    alpha = 0.05
    if p_value > alpha:
        print("Data looks Gaussian (fail to reject H0)") #normal distribution
    else:
        print("Data does not look Gaussian (reject H0)")

# %%
# unpaired t-test

from scipy.stats import ttest_ind, ttest_ind_from_stats
from scipy.special import stdtr


output = ttest_ind(group1, group2, nan_policy= "omit") #ttest_ind = from independent groups; 'omit' ignores NaN

print(output)

# %%
