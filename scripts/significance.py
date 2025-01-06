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
"""PUBLISHABLE"""
#Feeding experiment 12h intervall with 3 replicates and contr (no food), T, D

import pandas as pd
from scipy.stats import fisher_exact, chi2_contingency
from statsmodels.stats.multitest import multipletests

# Load the dataset
file_path = "/Users/nadine/Documents/paper/Naomi-NS-maturation/feeding-12h/Feeding-exp-12h-steps.csv"
data = pd.read_csv(file_path)

# Initialize lists to store results
pairwise_results = []
replicate_variations = []

# Get unique combinations of 'Age' and pairwise 'Group' comparisons
ages = data["Age"].unique()
groups = data["Group"].unique()

# Analyze each replicate for variation
for age in ages:
    for group in groups:
        # Subset data for this group and age
        subset = data[(data["Group"] == group) & (data["Age"] == age)]
        
        # Collect replicate data
        contingency_table = subset[["filled gut", "empty gut"]].values
        
        # Filter out tables with rows or columns of all zeros
        if contingency_table.sum(axis=1).min() > 0 and contingency_table.sum(axis=0).min() > 0:
            # Test for variation across replicates using chi-squared test
            if contingency_table.shape[0] > 1:  # Ensure there are replicates
                _, replicate_p_value, _, _ = chi2_contingency(contingency_table)
                replicate_variations.append({
                    "Age": age,
                    "Group": group,
                    "Replicate Variation P-value": replicate_p_value
                })
        else:
            replicate_variations.append({
                "Age": age,
                "Group": group,
                "Replicate Variation P-value": None  # Mark as invalid due to zero rows/columns
            })

# Perform pairwise comparisons for each age
for age in ages:
    for i, group1 in enumerate(groups):
        for group2 in groups[i + 1:]:
            # Subset data for the two groups at the current age
            data1 = data[(data["Group"] == group1) & (data["Age"] == age)]
            data2 = data[(data["Group"] == group2) & (data["Age"] == age)]

            # Aggregate 'empty gut' and 'filled gut' counts across replicates
            count1 = [data1["filled gut"].sum(), data1["empty gut"].sum()]
            count2 = [data2["filled gut"].sum(), data2["empty gut"].sum()]

            # Create a contingency table
            contingency_table = [count1, count2]

            # Perform Fisher's exact test
            odds_ratio, p_value = fisher_exact(contingency_table)

            # Store the result
            pairwise_results.append({
                "Age": age,
                "Group1": group1,
                "Group2": group2,
                "Odds Ratio": odds_ratio,
                "P-value": p_value
            })

# Convert replicate variation results to a DataFrame
replicate_variation_df = pd.DataFrame(replicate_variations)

# Convert pairwise results to a DataFrame
pairwise_results_df = pd.DataFrame(pairwise_results)

# Apply Bonferroni correction
pairwise_results_df["Adjusted P-value (Bonferroni)"] = multipletests(
    pairwise_results_df["P-value"], method="bonferroni"
)[1]

# Apply Holm-Bonferroni correction
pairwise_results_df["Adjusted P-value (Holm)"] = multipletests(
    pairwise_results_df["P-value"], method="holm"
)[1]

# Display all pairwise results with both corrections
print("Replicate Variation Results:")
print(replicate_variation_df)

print("All Pairwise Results with P-values and Adjusted P-values:")
print(pairwise_results_df)

# Save results to separate CSV files
replicate_variation_df.to_csv("/Users/nadine/Documents/paper/Naomi-NS-maturation/feeding-12h/replicate_variation_results.csv", index=False)
pairwise_results_df[["Age", "Group1", "Group2", "Odds Ratio", "P-value", "Adjusted P-value (Bonferroni)"]].to_csv(
    "/Users/nadine/Documents/paper/Naomi-NS-maturation/feeding-12h/pairwise_results_bonferroni.csv", index=False
)
pairwise_results_df[["Age", "Group1", "Group2", "Odds Ratio", "P-value", "Adjusted P-value (Holm)"]].to_csv(
    "/Users/nadine/Documents/paper/Naomi-NS-maturation/feeding-12h/pairwise_results_holm.csv", index=False
)

print("\nResults saved to 'replicate_variation_results.csv', 'pairwise_results_bonferroni.csv' and 'pairwise_results_holm.csv'")

# %%
"""PUBLISHABLE"""
#Yolk content from EdU larvae with 3 replicates and contr (no food), T, D

import pandas as pd
from scipy.stats import fisher_exact, chi2_contingency
from statsmodels.stats.multitest import multipletests

# Load the dataset
file_path = "/Users/nadine/Documents/paper/Naomi-NS-maturation/yolk-EdU-larvae.csv"
data = pd.read_csv(file_path)

# Initialize lists to store results
pairwise_results = []
replicate_variations = []

# Get unique combinations of 'Age' and pairwise 'Group' comparisons
ages = data["Age"].unique()
groups = data["Group"].unique()

# Analyze each replicate for variation
for age in ages:
    for group in groups:
        # Subset data for this group and age
        subset = data[(data["Group"] == group) & (data["Age"] == age)]
        
        # Collect replicate data
        contingency_table = subset[["no yolk", "yolk"]].values
        
        # Filter out tables with rows or columns of all zeros
        if contingency_table.sum(axis=1).min() > 0 and contingency_table.sum(axis=0).min() > 0:
            # Test for variation across replicates using chi-squared test
            if contingency_table.shape[0] > 1:  # Ensure there are replicates
                _, replicate_p_value, _, _ = chi2_contingency(contingency_table)
                replicate_variations.append({
                    "Age": age,
                    "Group": group,
                    "Replicate Variation P-value": replicate_p_value
                })
        else:
            replicate_variations.append({
                "Age": age,
                "Group": group,
                "Replicate Variation P-value": None  # Mark as invalid due to zero rows/columns
            })

# Perform pairwise comparisons for each age
for age in ages:
    for i, group1 in enumerate(groups):
        for group2 in groups[i + 1:]:
            # Subset data for the two groups at the current age
            data1 = data[(data["Group"] == group1) & (data["Age"] == age)]
            data2 = data[(data["Group"] == group2) & (data["Age"] == age)]

            # Aggregate 'empty gut' and 'filled gut' counts across replicates
            count1 = [data1["no yolk"].sum(), data1["yolk"].sum()]
            count2 = [data2["no yolk"].sum(), data2["yolk"].sum()]

            # Create a contingency table
            contingency_table = [count1, count2]

            # Perform Fisher's exact test
            odds_ratio, p_value = fisher_exact(contingency_table)

            # Store the result
            pairwise_results.append({
                "Age": age,
                "Group1": group1,
                "Group2": group2,
                "Odds Ratio": odds_ratio,
                "P-value": p_value
            })

# Convert replicate variation results to a DataFrame
replicate_variation_df = pd.DataFrame(replicate_variations)

# Convert pairwise results to a DataFrame
pairwise_results_df = pd.DataFrame(pairwise_results)

# Apply Bonferroni correction
pairwise_results_df["Adjusted P-value (Bonferroni)"] = multipletests(
    pairwise_results_df["P-value"], method="bonferroni"
)[1]

# Apply Holm-Bonferroni correction
pairwise_results_df["Adjusted P-value (Holm)"] = multipletests(
    pairwise_results_df["P-value"], method="holm"
)[1]

# Display all pairwise results with both corrections
print("Replicate Variation Results:")
print(replicate_variation_df)

print("All Pairwise Results with P-values and Adjusted P-values:")
print(pairwise_results_df)

# Save results to separate CSV files
replicate_variation_df.to_csv("/Users/nadine/Documents/paper/Naomi-NS-maturation/replicate_variation_results_yolk.csv", index=False)
pairwise_results_df[["Age", "Group1", "Group2", "Odds Ratio", "P-value", "Adjusted P-value (Bonferroni)"]].to_csv(
    "/Users/nadine/Documents/paper/Naomi-NS-maturation/pairwise_results_bonferroni_yolk.csv", index=False
)
pairwise_results_df[["Age", "Group1", "Group2", "Odds Ratio", "P-value", "Adjusted P-value (Holm)"]].to_csv(
    "/Users/nadine/Documents/paper/Naomi-NS-maturation/pairwise_results_holm_yolk.csv", index=False
)

print("\nResults saved to 'replicate_variation_results.csv', 'pairwise_results_bonferroni.csv' and 'pairwise_results_holm.csv'")



# %%
