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
column_names = ['3dpf Tetraselmis']
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

# Mann-Whitney U test (non-parametric test also known as the 
# Wilcoxon rank-sum test, is a non-parametric test used to assess 
# whether two independent samples come from the same population or 
# have the same median. It's particularly useful when the assumptions 
# of the t-test (such as normality and equal variances) are not met.

# Specify the column names for the two groups

all_column_names = ['3dpf cntr', '3dpf Tetraselmis' ,'3dpf G. marina' ,'4dpf cntr' ,'4dpf Tetraselmis' ,'4dpf G. marina' ,'5dpf cntr' ,'5dpf Tetraselmis' ,'5dpf G. marina' ,'6dpf cntr', '6dpf Tetraselmis', '6dpf G. marina', '7dpf cntr', '7dpf Tetraselmis', '7dpf G. marina' , '8dpf cntr' ,'8dpf Tetraselmis' ,'8dpf G. marina']  

group1_column_name = '3dpf cntr'
group2_column_name = '8dpf G. marina'

# Remove NaN values from each group column-wise
group1 = df[group1_column_name].dropna()
group2 = df[group2_column_name].dropna()

# Perform the Mann-Whitney U test
mannwhitneyu_statistic, mannwhitneyu_pvalue = stats.mannwhitneyu(group1, group2)

alpha = 0.05  # significance level
print("Mann-Whitney U test statistic:", mannwhitneyu_statistic)
print("p-value:", mannwhitneyu_pvalue)

if mannwhitneyu_pvalue < alpha:
    print("Reject the null hypothesis. There are significant differences between the groups.")
else:
    print("Fail to reject the null hypothesis. There are no significant differences between the groups.")
# %%
# unpaired t-test

from scipy.stats import ttest_ind, ttest_ind_from_stats
from scipy.special import stdtr


output = ttest_ind(group1, group2, nan_policy= "omit") #ttest_ind = from independent groups; 'omit' ignores NaN

print(output)

# %%
