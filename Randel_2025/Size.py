"""Normal distributed data (Shapiro-Wilk test)"""
"""Assess Replicate Variation using one-way ANOVA.
Compare Groups using one-way ANOVA and Tukey's HSD for pairwise comparisons.
Compare Across Ages using repeated-measures ANOVA (if replicates are consistent across ages).
Multiple Comparisons using Holm-Bonferroni correction."""

#%%
import pandas as pd
from scipy.stats import shapiro, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

# Load the dataset
file_path = "/path/name.csv"
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
pd.DataFrame(normality_results).to_csv("/path/normality_results.csv", index=False)
pd.DataFrame(replicate_variation_results).to_csv("/path/replicate_variation_results.csv", index=False)
pd.DataFrame(group_comparisons).to_csv("/path/group_comparisons_results.csv", index=False)
pd.DataFrame(age_comparisons).to_csv("/path/age_comparisons_results.csv", index=False)

# %%
"""Pairwise comparison (pairwise t-test) after Anova"""

from itertools import combinations
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import pandas as pd

# Load the dataset
file_path = "/path/name.csv"
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
pairwise_group_results_df.to_csv("/path/pairwise_ttest_group_comparisons.csv", index=False)

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
pairwise_age_results_df.to_csv("/path/pairwise_ttest_age_comparisons.csv", index=False)