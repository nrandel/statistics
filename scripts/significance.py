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
"""PUBLISHABLE"""
#Size from EdU larvae with 3 replicates and contr (no food), T, D

#Test normal distributionn with  Shapiro-Wilk test
#(If p>0.05, than data are normally distributed) 

import pandas as pd
from scipy.stats import shapiro

# Load the dataset
file_path = "/Users/nadine/Documents/paper/Naomi-NS-maturation/size-EdU-larvae.csv"
data = pd.read_csv(file_path)

# Initialize a list to store Shapiro-Wilk test results
shapiro_results = []

# Get unique combinations of Group, Age, and Replicate
grouped_data = data.groupby(["Group", "Age", "Replicate"])

# Perform Shapiro-Wilk test for each subset of data
for (group, age, replicate), subset in grouped_data:
    measurements = subset["Measurement"]
    
    # Ensure there are enough data points for the test (at least 3)
    if len(measurements) >= 3:
        stat, p_value = shapiro(measurements)
        normality = "Normal" if p_value > 0.05 else "Not Normal"
    else:
        stat, p_value = None, None
        normality = "Insufficient Data"
    
    # Append the result
    shapiro_results.append({
        "Group": group,
        "Age": age,
        "Replicate": replicate,
        "Shapiro-Wilk Statistic": stat,
        "P-value": p_value,
        "Normality": normality
    })

# Convert results to a DataFrame
shapiro_results_df = pd.DataFrame(shapiro_results)

# Save results to a CSV file
output_path = "/Users/nadine/Documents/paper/Naomi-NS-maturation/shapiro_wilk_results.csv"
shapiro_results_df.to_csv(output_path, index=False)

# Display the results
print("Shapiro-Wilk Test Results:")
print(shapiro_results_df)

print(f"\nResults saved to '{output_path}'")



# %%
"""PUBLISHABLE"""
#Size from EdU larvae with 3 replicates and contr (no food), T, D

#Normal distributed (parametric test)
"""Assess Replicate Variation using one-way ANOVA.
Compare Groups (Across No Food, T, D) using one-way ANOVA and Tukey's HSD for pairwise comparisons.
Compare Across Ages using repeated-measures ANOVA (if replicates are consistent across ages).
Multiple Comparisons using Holm-Bonferroni correction."""

import pandas as pd
from scipy.stats import shapiro, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

# Load the dataset
file_path = "/Users/nadine/Documents/paper/Naomi-NS-maturation/size-EdU-larvae.csv"
data = pd.read_csv(file_path)

# Initialize results storage
replicate_variation_results = []
group_comparisons = []
age_comparisons = []

# Get unique ages and groups
ages = data["Age"].unique()
groups = data["Group"].unique()

# 1. Check for normal distribution for each group and replicate
normality_results = []

for age in ages:
    for group in groups:
        for replicate in data[data["Group"] == group]["Replicate"].unique():
            subset = data[(data["Age"] == age) & (data["Group"] == group) & (data["Replicate"] == replicate)]
            if len(subset["Measurement"]) >= 3:  # Shapiro-Wilk requires at least 3 data points
                stat, p_value = shapiro(subset["Measurement"])
                normality_results.append({
                    "Age": age,
                    "Group": group,
                    "Replicate": replicate,
                    "W-statistic": stat,
                    "P-value": p_value,
                    "Normal Distribution": "Yes" if p_value > 0.05 else "No"
                })

# 2. Assess replicate variation using one-way ANOVA
for age in ages:
    for group in groups:
        subset = data[(data["Age"] == age) & (data["Group"] == group)]
        grouped = subset.groupby("Replicate")["Measurement"].apply(list)
        if all(len(g) > 1 for g in grouped):  # Ensure each replicate has data
            stat, p_value = f_oneway(*grouped)
            replicate_variation_results.append({
                "Age": age,
                "Group": group,
                "F-statistic": stat,
                "P-value": p_value
            })

# 3. Compare groups across no food, T, D
for age in ages:
    subset = data[data["Age"] == age]
    stat, p_value = f_oneway(
        *[subset[subset["Group"] == group]["Measurement"] for group in groups if len(subset[subset["Group"] == group]) > 1]
    )
    group_comparisons.append({
        "Age": age,
        "F-statistic": stat,
        "P-value": p_value
    })

# Apply Holm-Bonferroni correction for group comparisons
group_p_values = [result["P-value"] for result in group_comparisons]
adjusted_p_values = multipletests(group_p_values, method="holm")[1]

for i, result in enumerate(group_comparisons):
    result["Adjusted P-value"] = adjusted_p_values[i]

# 4. Compare across ages for each group
for group in groups:
    subset = data[data["Group"] == group]
    stat, p_value = f_oneway(
        *[subset[subset["Age"] == age]["Measurement"] for age in ages if len(subset[subset["Age"] == age]) > 1]
    )
    age_comparisons.append({
        "Group": group,
        "F-statistic": stat,
        "P-value": p_value
    })

# Apply Holm-Bonferroni correction for age comparisons
age_p_values = [result["P-value"] for result in age_comparisons]
adjusted_p_values_age = multipletests(age_p_values, method="holm")[1]

for i, result in enumerate(age_comparisons):
    result["Adjusted P-value"] = adjusted_p_values_age[i]

# Save results to CSV files
pd.DataFrame(normality_results).to_csv("/Users/nadine/Documents/paper/Naomi-NS-maturation/normality_results.csv", index=False)
pd.DataFrame(replicate_variation_results).to_csv("/Users/nadine/Documents/paper/Naomi-NS-maturation/replicate_variation_results.csv", index=False)
pd.DataFrame(group_comparisons).to_csv("/Users/nadine/Documents/paper/Naomi-NS-maturation/group_comparisons_results.csv", index=False)
pd.DataFrame(age_comparisons).to_csv("/Users/nadine/Documents/paper/Naomi-NS-maturation/age_comparisons_results.csv", index=False)

print("Results saved to 'normality_results.csv', 'replicate_variation_results.csv', 'group_comparisons_results.csv', and 'age_comparisons_results.csv'")


# %%
"""PUBLISHABLE"""
#pairwise comparison after Anova
from itertools import combinations
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import pandas as pd

# Load the dataset
file_path = "/Users/nadine/Documents/paper/Naomi-NS-maturation/size-EdU-larvae.csv"
data = pd.read_csv(file_path)

# Initialize storage for results
pairwise_group_results = []
pairwise_age_results = []

# 1. Pairwise t-tests for group comparisons within each age
for age in data["Age"].unique():
    subset = data[data["Age"] == age]
    groups = subset["Group"].unique()
    if len(groups) > 1:  # Need at least 2 groups to compare
        for group1, group2 in combinations(groups, 2):
            group1_data = subset[subset["Group"] == group1]["Measurement"]
            group2_data = subset[subset["Group"] == group2]["Measurement"]
            if len(group1_data) > 1 and len(group2_data) > 1:  # Require at least 2 data points per group
                t_stat, p_value = ttest_ind(group1_data, group2_data, equal_var=False)
                pairwise_group_results.append({
                    "Age": age,
                    "Group1": group1,
                    "Group2": group2,
                    "T-statistic": t_stat,
                    "P-value": p_value
                })

# Convert results to DataFrame
pairwise_group_results_df = pd.DataFrame(pairwise_group_results)

# Apply Holm-Bonferroni correction to group comparisons
group_p_values = pairwise_group_results_df["P-value"]
adjusted_p_values_group = multipletests(group_p_values, method="holm")[1]
pairwise_group_results_df["Adjusted P-value"] = adjusted_p_values_group

# Save results
pairwise_group_results_df.to_csv("/Users/nadine/Documents/paper/Naomi-NS-maturation/pairwise_ttest_group_comparisons.csv", index=False)

# 2. Pairwise t-tests for age comparisons within each group
for group in data["Group"].unique():
    subset = data[data["Group"] == group]
    ages = subset["Age"].unique()
    if len(ages) > 1:  # Need at least 2 ages to compare
        for age1, age2 in combinations(ages, 2):
            age1_data = subset[subset["Age"] == age1]["Measurement"]
            age2_data = subset[subset["Age"] == age2]["Measurement"]
            if len(age1_data) > 1 and len(age2_data) > 1:  # Require at least 2 data points per age
                t_stat, p_value = ttest_ind(age1_data, age2_data, equal_var=False)
                pairwise_age_results.append({
                    "Group": group,
                    "Age1": age1,
                    "Age2": age2,
                    "T-statistic": t_stat,
                    "P-value": p_value
                })

# Convert results to DataFrame
pairwise_age_results_df = pd.DataFrame(pairwise_age_results)

# Apply Holm-Bonferroni correction to age comparisons
age_p_values = pairwise_age_results_df["P-value"]
adjusted_p_values_age = multipletests(age_p_values, method="holm")[1]
pairwise_age_results_df["Adjusted P-value"] = adjusted_p_values_age

# Save results
pairwise_age_results_df.to_csv("/Users/nadine/Documents/paper/Naomi-NS-maturation/pairwise_ttest_age_comparisons.csv", index=False)

print("Pairwise t-test comparisons saved to 'pairwise_ttest_group_comparisons.csv' and 'pairwise_ttest_age_comparisons.csv'.")

# %%
