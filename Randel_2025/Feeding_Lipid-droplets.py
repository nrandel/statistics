"""Fisher’s exact test: Variation between the biological replicates and a significant association between the variables.""" 
"""Holm-Bonferroni correction: Multiple-comparison analysis."""
#%%
import pandas as pd
from scipy.stats import fisher_exact, chi2_contingency
from statsmodels.stats.multitest import multipletests

#%%
"""Feeding experiment 12h intervall with 3 replicates"""

# Load the dataset
file_path = "/path/name.csv"
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

# Apply Holm-Bonferroni correction
pairwise_results_df["Adjusted P-value (Holm)"] = multipletests(
    pairwise_results_df["P-value"], method="holm"
)[1]

# Display all pairwise results with both corrections
print("Replicate Variation Results:")
print(replicate_variation_df)

print("All Pairwise Results with P-values and Adjusted P-values:")
print(pairwise_results_df)

#%%
"""Lipid droplet content from EdU larvae with 3 replicates and contr"""

import pandas as pd
from scipy.stats import fisher_exact, chi2_contingency
from statsmodels.stats.multitest import multipletests

# Load the dataset
file_path = "/path/name.csv"
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

# Apply Holm-Bonferroni correction
pairwise_results_df["Adjusted P-value (Holm)"] = multipletests(
    pairwise_results_df["P-value"], method="holm"
)[1]

# Display all pairwise results with both corrections
print("Replicate Variation Results:")
print(replicate_variation_df)

print("All Pairwise Results with P-values and Adjusted P-values:")
print(pairwise_results_df)
