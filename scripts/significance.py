# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import csv

# %%
# Import csv as panda dataframe 

# EdU exp
#df = pd.read_csv("/Users/nadine/Documents/paper/Naomi-NS-maturation/BF_EdU-larvae/BF_EdU-larvae_size-merge_um.csv")  

# Feeding exp
df = pd.read_csv("/Users/nadine/Documents/paper/Naomi-NS-maturation/Feeding-Experiments-AxioZoom_Exeter_1_7-13-2023/Feeding-size_um.csv")  

#%% Choose columnns by name
#column_names = ['3dpf cntr', '3dpf Tetraselmis' ,'3dpf G. marina' ,'4dpf cntr' ,'4dpf Tetraselmis' ,'4dpf G. marina' ,'5dpf cntr' ,'5dpf Tetraselmis' ,'5dpf G. marina' ,'6dpf cntr', '6dpf Tetraselmis', '6dpf G. marina', '7dpf cntr', '7dpf Tetraselmis', '7dpf G. marina' , '8dpf cntr' ,'8dpf Tetraselmis' ,'8dpf G. marina']  
column_names = ['3dpf Tetraselmis' ,'3dpf G. marina' ,'4dpf Tetraselmis' ,'4dpf G. marina' ,'5dpf Tetraselmis' ,'5dpf G. marina', '6dpf Tetraselmis', '6dpf G. marina']
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

group1_column_name = '6dpf Tetraselmis'
group2_column_name = '6dpf G. marina'

# Remove NaN values from each group column-wise
group1 = df[group1_column_name].dropna()
group2 = df[group2_column_name].dropna()

# Perform the Mann-Whitney U test
mannwhitneyu_statistic, mannwhitneyu_pvalue = stats.mannwhitneyu(group1, group2)

alpha = 0.05  # significance level
print("Mann-Whitney U test statistic:", mannwhitneyu_statistic)
print(group1_column_name, group2_column_name, "p-value:", mannwhitneyu_pvalue)

if mannwhitneyu_pvalue < alpha:
    print("Reject the null hypothesis. There are significant differences between the groups.")
else:
    print("Fail to reject the null hypothesis. There are no significant differences between the groups.")

# %%
# unpaired t-test

# Specify the column names for the two groups
all_column_names = ['3dpf cntr', '3dpf Tetraselmis', '3dpf G. marina', '4dpf cntr', '4dpf Tetraselmis', '4dpf G. marina', '5dpf cntr', '5dpf Tetraselmis', '5dpf G. marina', '6dpf cntr', '6dpf Tetraselmis', '6dpf G. marina', '7dpf cntr', '7dpf Tetraselmis', '7dpf G. marina', '8dpf cntr', '8dpf Tetraselmis', '8dpf G. marina']

group1_column_name = '5dpf G. marina'
group2_column_name = '5dpf cntr'

# Extract the actual data for the two groups and remove NaN values
group1_data = df[group1_column_name].dropna()
group2_data = df[group2_column_name].dropna()

# Perform the t-test
t_statistic, p_value = stats.ttest_ind(group1_data, group2_data)

alpha = 0.05  # significance level
print("t-statistic:", t_statistic)
print("p-value:", p_value)

if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference between the means.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference between the means.")

# %%
#chi-square test of yolk data

import numpy as np
from scipy.stats import chi2_contingency

# Example data (replace with your actual data)
# Each row represents a condition, and each column represents the presence (1) or absence (' ') of the feature
data = np.array([[10, 10],  # Condition 1 'no food': 10 individuals with feature, 10 without
                 [2, 20],  # Condition 2 'T': 2 individuals with feature, 20 without
                 [5, 19]])  # Condition 3 'D': 5 individuals with feature, 19 without

# Perform Chi-squared test
chi2_stat, p_val, dof, expected = chi2_contingency(data)

# Output results
print("Chi-squared statistic:", chi2_stat)
print("P-value:", p_val)
print("Degrees of freedom:", dof)
print("Expected frequencies:")
print(expected)

# Interpret results
alpha = 0.05  # Significance level
print("\n")
if p_val < alpha:
    print("There is a significant association between condition and feature presence.")
else:
    print("There is no significant association between condition and feature presence.")

# %%
import numpy as np
from scipy.stats import chi2_contingency

# Example data (replace with your actual data)
data = np.array([[10, 10],  # Condition 1: 10 individuals with feature, 10 without
                 [2, 20]]) # Condition 2: 15 individuals with feature, 25 without

# Perform Chi-squared test for independence
chi2_stat, p_val, dof, expected = chi2_contingency(data)

# Output results
print("Chi-squared statistic:", chi2_stat)
print("P-value:", p_val)
print("Degrees of freedom:", dof)
print("Expected frequencies:")
print(expected)

# Interpret results
alpha = 0.05  # Significance level
if p_val < alpha:
    print("There is a significant association between condition and feature presence.")
else:
    print("There is no significant association between condition and feature presence.")




# %%
# Fisher's exact test instead of chi2-contincency for small (<=5) expected frequencies and conditio == 0
# The input table must be 2x2

import numpy as np
from scipy.stats import fisher_exact

# Define the data
data = np.array([[9, 10],  # Condition 1 'no food': 10 individuals with feature, 10 without
                 [19, 5]])   # Condition 2 'T': 2 individuals with feature, 20 without  

# Perform Fisher's exact test
odds_ratio, p_value = fisher_exact(data)

print("Odds Ratio:", odds_ratio)
print("P-value:", p_value)

# %%
