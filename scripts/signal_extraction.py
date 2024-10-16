# %%
import glob
import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import csv

# %%
# Calculate SUM and the weighted average or total %Area as a fraction 
# of the entire 3D stack.


# Define the main directory containing the sample folders
main_directory = "/Users/nadine/Documents/paper/Naomi-NS-maturation/cLM_EdU-larvae/analysis/ROIs/"

# Initialize a list to store the results of all samples
all_results = []

def weighted_avg(df, val_col, weight_col):
    return (df[val_col] * df[weight_col]).sum() / df[weight_col].sum()

# Loop through each sample folder in the main directory
for sample_folder in os.listdir(main_directory):
    sample_path = os.path.join(main_directory, sample_folder)
    
    # Only proceed if it's a directory
    if os.path.isdir(sample_path):
        # Find all CSV files within the sample folder
        csv_files = glob.glob(os.path.join(sample_path, "**", "*.csv"), recursive=True)

        # Initialize a list to store DataFrames for the current sample
        dfs = []

        # Loop through each CSV file within the sample folder
        for file in csv_files:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file)
            
            # Add a new column to store the filename and sample folder
            df['filename'] = os.path.basename(file)
            df['sample'] = sample_folder  # Add sample identifier
            
            # Append the DataFrame to the list
            dfs.append(df)

        if dfs:  # Check if there are DataFrames
            # Combine all DataFrames for this sample
            combined_df = pd.concat(dfs, ignore_index=True)

            # Step 1: Add a new column for signal area for each slice
            combined_df['Signal_Area'] = (combined_df['%Area'] / 100) * combined_df['Area']

            # Step 2: Group by filename and calculate total area, total signal area
            sample_results = combined_df.groupby('filename').agg(
                total_area=('Area', 'sum'),                      
                total_signal_area=('Signal_Area', 'sum')
            ).reset_index()

            # Calculate weighted average %Area separately
            sample_results['weighted_avg_percent_area'] = combined_df.groupby('filename').apply(
                lambda x: weighted_avg(x, '%Area', 'Area')
            ).values

            # Add the sample identifier to the results
            sample_results['sample'] = sample_folder
            
            # Append the sample's results to the main list
            all_results.append(sample_results)

# Step 3: Combine results for all samples
final_results = pd.concat(all_results, ignore_index=True)

# Display the final result
print(final_results[['sample', 'filename', 'total_area', 'total_signal_area', 'weighted_avg_percent_area']])

# %%

# Parse the data for area, age_group,sample_group
# Assuming final_results is already defined from your previous code

# Step 1: Function to extract area, age group, and sample group
def extract_grouping_info(row):
    filename = row['filename']
    sample = row['sample']
    
    # Extract area from filename (the part before the first underscore)
    area = filename.split('_')[0]  # Get the first part before the first underscore
    
    # Extract age group from sample (the part after 'Pd_' and before 'dpf')
    age_group = sample.split('_')[1]  # This takes the second part (e.g., '3dpfdpf')
    
    # Shorten age_group to just the number
    age_group = age_group[:-3]  # Reduce '3dpfdpf' to '3'
    
    # Extract sample group (D, T, or C) without the number
    sample_group = sample.split('_')[2][0]  # Get the first letter (C, D, or T)
    
    return area, age_group, sample_group

# Step 2: Apply the function to extract grouping information and create new columns
final_results['area'], final_results['age_group'], final_results['sample_group'] = zip(
    *final_results.apply(extract_grouping_info, axis=1)
)

# Print extracted columns to check correctness, along with all columns from final_results
print("\nSample of Extracted Grouping Info:")
print(final_results[['filename', 'area', 'age_group', 'sample_group'] + final_results.columns.tolist()])

# Step 3: Group by age group, sample group, and area
grouped_data = final_results.groupby(['age_group', 'sample_group', 'area']).agg(
    total_area=('total_area', 'sum'),
    total_signal_area=('total_signal_area', 'sum')
).reset_index()

# Step 4: Check if groups are being formed correctly
if grouped_data.empty:
    print("No groups were formed. Please check the grouping logic.")
else:
    print(f"\nNumber of groups formed: {grouped_data.shape[0]}")
    print("Example groups:")
    for _, group_df in grouped_data.iterrows():
        print(f"Group: {group_df['age_group']} {group_df['sample_group']} {group_df['area']}")
        print(f"Total Area: {group_df['total_area']}, Total Signal Area: {group_df['total_signal_area']}")

# Optional: Print the grouped data for inspection
print("\nGrouped Data:")
print(grouped_data)


# %%

# Strip plot for total signal area
# Assuming final_results is defined from your previous steps

# Set the age group you want to plot (change this to None to plot all age groups)
selected_age_group = None  # Change this to a specific age group like '3dpf' or keep as None

# Filter the data based on the selected age group
if selected_age_group:
    age_groups = [selected_age_group]
else:
    age_groups = final_results['age_group'].unique()

# Create a figure with subplots
num_age_groups = len(age_groups)
fig, axes = plt.subplots(nrows=1, ncols=num_age_groups, figsize=(10, 5), sharey=True)

# If there's only one age group, axes will not be a list, so we need to handle that case
if num_age_groups == 1:
    axes = [axes]  # Convert axes to a list for consistency

# Loop through each age group to create a subplot
for ax, age in zip(axes, age_groups):
    # Filter the data for the current age group
    age_group_data = final_results[final_results['age_group'] == age]
    
    # Create a strip plot for the current age group
    sns.stripplot(data=age_group_data, x='sample_group', y='total_signal_area', 
                  hue='area', dodge=True, marker='o', alpha=0.7, ax=ax)
    
    # Set title and labels
    ax.set_title(f'Age Group: {age}')
    ax.set_xlabel('Sample Group')
    ax.set_ylabel('Total Signal Area')

# Adjust layout
plt.tight_layout()

# Show the legend
plt.legend(title='Area', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.show()


# %%
# Strip plot for weighted_avg_percent_area
# Assuming final_results is defined from your previous steps

# Set the age group you want to plot (change this to None to plot all age groups)
selected_age_group = None  # Change this to a specific age group like '3dpf' or keep as None

# Filter the data based on the selected age group
if selected_age_group:
    age_groups = [selected_age_group]
else:
    age_groups = final_results['age_group'].unique()

# Create a figure with subplots
num_age_groups = len(age_groups)
fig, axes = plt.subplots(nrows=1, ncols=num_age_groups, figsize=(10, 5), sharey=True)

# If there's only one age group, axes will not be a list, so we need to handle that case
if num_age_groups == 1:
    axes = [axes]  # Convert axes to a list for consistency

# Loop through each age group to create a subplot
for ax, age in zip(axes, age_groups):
    # Filter the data for the current age group
    age_group_data = final_results[final_results['age_group'] == age]
    
    # Create a strip plot for the current age group
    sns.stripplot(data=age_group_data, x='sample_group', y='weighted_avg_percent_area', 
                  hue='area', dodge=True, marker='o', alpha=0.7, ax=ax)
    
    # Set title and labels
    ax.set_title(f'Age Group: {age}')
    ax.set_xlabel('Sample Group')
    ax.set_ylabel('weighted_avg_percent_area')

# Adjust layout
plt.tight_layout()

# Show the legend
plt.legend(title='Area', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.show()
# %%
###TEST###
###REFINED PLOT FKT###

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming final_results is defined from your previous steps

# Step 1: Customize the order of sample groups (C, T, D, etc.)
custom_sample_group_order = ['C', 'T', 'D']  # Change this to your desired order

# Step 2: Customize the order of areas (Head, 1st, 2nd, etc.)
custom_area_order = ['Head', '1st', '2nd', '3rd', 'Pyg']  # Adjust based on your areas

# Step 3: Plot for each age group separately with custom y-axis scaling
age_groups = final_results['age_group'].unique()

for age in age_groups:
    # Filter the data for the current age group
    age_group_data = final_results[final_results['age_group'] == age]
    
    # Sort sample_group and area columns based on the custom orders
    age_group_data['sample_group'] = pd.Categorical(age_group_data['sample_group'], categories=custom_sample_group_order, ordered=True)
    age_group_data['area'] = pd.Categorical(age_group_data['area'], categories=custom_area_order, ordered=True)

    # Create a new figure for each age group
    plt.figure(figsize=(8, 6))
    
    # Create a strip plot
    sns.stripplot(data=age_group_data, x='sample_group', y='total_signal_area', 
                  hue='area', dodge=True, marker='o', alpha=0.7, size=10, palette='Set1')
    
    # Set title and labels
    plt.title(f'Strip Plot of Total Signal Area for Age Group: {age}')
    plt.xlabel('Sample Group')
    plt.ylabel('Total Signal Area')

    # Show the legend outside the plot
    plt.legend(title='Area', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust y-axis scale to fit the data optimally
    plt.ylim(age_group_data['total_signal_area'].min() - 100, age_group_data['total_signal_area'].max() + 1000)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

# %%
