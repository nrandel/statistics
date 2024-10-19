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

# Step 1: Function to extract area, age group, sample group, and replicate
def extract_grouping_info(row):
    filename = row['filename']
    sample = row['sample']
    
    # Extract area from filename (the part before the first underscore)
    area = filename.split('_')[0]  # Get the first part before the first underscore
    
    # Extract age group from sample (the part after 'Pd_' and before 'dpf')
    age_group = sample.split('_')[1]  # This takes the second part (e.g., '3dpf')
    
    # Shorten age_group to just the number
    age_group = age_group[:-3]  # Reduce '3dpfdpf' to '3'
    
    # Extract sample group (D, T, or C) without the number
    sample_group = sample.split('_')[2][0]  # Get the first letter (C, D, or T)
    
    # Extract replicate number from sample (the number after the first letter in the sample group)
    replicate = sample.split('_')[2][1:].split('-')[0]  # Extract the number after T, C, or D
    
    return area, age_group, sample_group, replicate

# Step 2: Apply the function to extract grouping information and create new columns
final_results['area'], final_results['age_group'], final_results['sample_group'], final_results['replicate'] = zip(
    *final_results.apply(extract_grouping_info, axis=1)
)

# Print extracted columns to check correctness, along with all columns from final_results
print("\nSample of Extracted Grouping Info:")
print(final_results[['filename', 'area', 'age_group', 'sample_group', 'replicate']])

# Step 3: Group by age group, sample group, area, and replicate
grouped_data = final_results.groupby(['age_group', 'sample_group', 'area', 'replicate']).agg(
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
        print(f"Group: {group_df['age_group']} {group_df['sample_group']} {group_df['area']} Replicate: {group_df['replicate']}")
        print(f"Total Area: {group_df['total_area']}, Total Signal Area: {group_df['total_signal_area']}")

# Optional: Print the grouped data for inspection
print("\nGrouped Data:")
print(grouped_data)

# %%
#PLOTTING from grouped_data
# Assuming final_results is already defined in your environment

# Set the age group you want to plot (change this to None to plot all age groups)
selected_age_group = None  # Change this to a specific age group like '3dpf' or keep as None

# Step 3: Define the custom order of age groups
custom_age_group_order = ['3', '4', '5', '6', '7', '8']  # Adjust based on your dataset

# Filter the data based on the selected age group
if selected_age_group:
    age_groups = [selected_age_group]
else:
    # Filter the age groups based on the custom order
    age_groups = [age for age in custom_age_group_order if age in final_results['age_group'].unique()]

# Step 1: Customize the order of sample groups (C, T, D, etc.)
custom_sample_group_order = ['C', 'T', 'D']  # Change this to your desired order

# Step 2: Customize the order of areas (Head, 1st, 2nd, 3rd, Pyg)
custom_area_order = ['Head', '1st', '2nd', '3rd', 'Pyg']  # Adjust based on your areas

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
    
    # Create a strip plot for the current age group with customized orders
    sns.stripplot(data=age_group_data, x='sample_group', y='total_signal_area', 
                  hue='area', dodge=True, marker='o', alpha=0.7, ax=ax, 
                  order=custom_sample_group_order, hue_order=custom_area_order)
    
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
# PLOTTING from grouped_data
# Single plots and adjustment for y axis. plot per day
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

# Generate new dataframe with the 
# # sum all areas, sum 1st and 2nd, sum of 3rd and pyg, head, pyg, segments only 
# for each age_group, each sample_group, and each replicate


# Step 1: Sum all areas for each age_group, sample_group, and replicate
sum_all_areas = grouped_data.groupby(['age_group', 'sample_group', 'replicate']).agg(
    total_signal_all_areas=('total_signal_area', 'sum')
).reset_index()

# Step 2: Filter and sum only the '1st' and '2nd' areas for each age_group, sample_group, and replicate
sum_1st_2nd_areas = grouped_data[grouped_data['area'].isin(['1st', '2nd'])].groupby(
    ['age_group', 'sample_group', 'replicate']
).agg(
    total_signal_1st_2nd=('total_signal_area', 'sum')
).reset_index()

# Step 3: Filter and sum the '3rd' and 'Pyg' areas
sum_3rd_pyg_areas = grouped_data[grouped_data['area'].isin(['3rd', 'Pyg'])].groupby(
    ['age_group', 'sample_group', 'replicate']
).agg(
    total_signal_3rd_pyg=('total_signal_area', 'sum')
).reset_index()

# Step 4: Get total area for 'Head' area
total_head_area = grouped_data[grouped_data['area'] == 'Head'].groupby(
    ['age_group', 'sample_group', 'replicate']
).agg(
    total_signal_head=('total_signal_area', 'sum')
).reset_index()

# Step 5: Get total area for individual regions '1st', '2nd', '3rd', and 'Pyg'
total_1st_area = grouped_data[grouped_data['area'] == '1st'].groupby(
    ['age_group', 'sample_group', 'replicate']
).agg(
    total_signal_1st=('total_signal_area', 'sum')
).reset_index()

total_2nd_area = grouped_data[grouped_data['area'] == '2nd'].groupby(
    ['age_group', 'sample_group', 'replicate']
).agg(
    total_signal_2nd=('total_signal_area', 'sum')
).reset_index()

total_3rd_area = grouped_data[grouped_data['area'] == '3rd'].groupby(
    ['age_group', 'sample_group', 'replicate']
).agg(
    total_signal_3rd=('total_signal_area', 'sum')
).reset_index()

total_pyg_area = grouped_data[grouped_data['area'] == 'Pyg'].groupby(
    ['age_group', 'sample_group', 'replicate']
).agg(
    total_signal_pyg=('total_signal_area', 'sum')
).reset_index()

# Step 6: Merge all the resulting DataFrames into one
merged_data = pd.merge(sum_all_areas, sum_1st_2nd_areas, 
                       on=['age_group', 'sample_group', 'replicate'], 
                       how='left')
merged_data = pd.merge(merged_data, sum_3rd_pyg_areas, 
                       on=['age_group', 'sample_group', 'replicate'], 
                       how='left')
merged_data = pd.merge(merged_data, total_head_area, 
                       on=['age_group', 'sample_group', 'replicate'], 
                       how='left')
merged_data = pd.merge(merged_data, total_1st_area, 
                       on=['age_group', 'sample_group', 'replicate'], 
                       how='left')
merged_data = pd.merge(merged_data, total_2nd_area, 
                       on=['age_group', 'sample_group', 'replicate'], 
                       how='left')
merged_data = pd.merge(merged_data, total_3rd_area, 
                       on=['age_group', 'sample_group', 'replicate'], 
                       how='left')
merged_data = pd.merge(merged_data, total_pyg_area, 
                       on=['age_group', 'sample_group', 'replicate'], 
                       how='left')

# Step 7: Fill any missing values in the newly merged columns with 0 (if some areas were not present)
merged_data['total_signal_1st_2nd'].fillna(0, inplace=True)
merged_data['total_signal_3rd_pyg'].fillna(0, inplace=True)
merged_data['total_signal_head'].fillna(0, inplace=True)
merged_data['total_signal_1st'].fillna(0, inplace=True)
merged_data['total_signal_2nd'].fillna(0, inplace=True)
merged_data['total_signal_3rd'].fillna(0, inplace=True)
merged_data['total_signal_pyg'].fillna(0, inplace=True)

# Optional: Print the new merged data
print("\nNew DataFrame with Summed Areas:")
print(merged_data)



# %%
#PLOTTING from merged data
# Use merged_data instead of final_results
# Set the age group you want to plot (change this to None to plot all age groups)
selected_age_group = None  # Change this to a specific age group like '3dpf' or keep as None

# Step 3: Define the custom order of age groups
custom_age_group_order = ['3', '4', '5', '6', '7', '8']  # Adjust based on your dataset

# Filter the data based on the selected age group
if selected_age_group:
    age_groups = [selected_age_group]
else:
    # Filter the age groups based on the custom order
    age_groups = [age for age in custom_age_group_order if age in merged_data['age_group'].unique()]

# Step 1: Customize the order of sample groups (C, T, D, etc.)
custom_sample_group_order = ['C', 'T', 'D']  # Change this to your desired order

# Step 2: Customize the order of areas (Head, 1st, 2nd, 3rd, Pyg)
custom_area_order = ['total_signal_head', 'total_signal_1st_2nd', 'total_signal_3rd_pyg', 'total_signal_all_areas']  # Adjust based on your areas

# Create a figure with subplots
num_age_groups = len(age_groups)
fig, axes = plt.subplots(nrows=1, ncols=num_age_groups, figsize=(10, 5), sharey=True)

# If there's only one age group, axes will not be a list, so we need to handle that case
if num_age_groups == 1:
    axes = [axes]  # Convert axes to a list for consistency

# Loop through each age group to create a subplot
for ax, age in zip(axes, age_groups):
    # Filter the data for the current age group
    age_group_data = merged_data[merged_data['age_group'] == age]
    
    # Melt the data to create long format (so that columns can be plotted as values in 'area')
    melted_data = pd.melt(age_group_data, id_vars=['age_group', 'sample_group', 'replicate'], 
                          value_vars=['total_signal_head', 'total_signal_1st_2nd', 'total_signal_3rd_pyg', 'total_signal_all_areas'],
                          var_name='area', value_name='total_signal_area')

    # Create a strip plot for the current age group with customized orders
    sns.stripplot(data=melted_data, x='sample_group', y='total_signal_area', 
                  hue='area', dodge=True, marker='o', alpha=0.7, ax=ax, 
                  order=custom_sample_group_order, hue_order=custom_area_order)
    
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

# PLOTTING from merged_data
# Single plots and adjustment for y axis. plot per day

# Step 1: Customize the order of sample groups (C, T, D, etc.)
custom_sample_group_order = ['C', 'T', 'D']  # Change this to your desired order

# Step 2: Customize the order of areas (total_signal_head, total_signal_1st_2nd, etc.)
custom_area_order = ['total_signal_head', 'total_signal_1st_2nd', 'total_signal_3rd_pyg', 'total_signal_all_areas']  # Adjust based on your areas

# Step 3: Plot for each age group separately with custom y-axis scaling
age_groups = merged_data['age_group'].unique()

for age in age_groups:
    # Filter the data for the current age group
    age_group_data = merged_data[merged_data['age_group'] == age]
    
    # Melt the data to create long format for plotting (so that columns become values under 'area')
    melted_data = pd.melt(age_group_data, id_vars=['age_group', 'sample_group', 'replicate'], 
                          value_vars=['total_signal_head', 'total_signal_1st_2nd', 'total_signal_3rd_pyg', 'total_signal_all_areas'],
                          var_name='area', value_name='total_signal_area')

    # Sort sample_group and area columns based on the custom orders
    melted_data['sample_group'] = pd.Categorical(melted_data['sample_group'], categories=custom_sample_group_order, ordered=True)
    melted_data['area'] = pd.Categorical(melted_data['area'], categories=custom_area_order, ordered=True)

    # Create a new figure for each age group
    plt.figure(figsize=(8, 6))
    
    # Create a strip plot
    sns.stripplot(data=melted_data, x='sample_group', y='total_signal_area', 
                  hue='area', dodge=True, marker='o', alpha=0.7, size=10, palette='Set1')
    
    # Set title and labels
    plt.title(f'Strip Plot of Total Signal Area for Age Group: {age}')
    plt.xlabel('Sample Group')
    plt.ylabel('Total Signal Area')

    # Show the legend outside the plot
    plt.legend(title='Area', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust y-axis scale to fit the data optimally
    plt.ylim(melted_data['total_signal_area'].min() - 100, melted_data['total_signal_area'].max() + 1000)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


# %%
###TEST###
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assuming merged_data is already defined and structured as discussed

# Step 1: Create a figure for plotting
plt.figure(figsize=(10, 6))

# Step 2: Define the custom order for sample groups
custom_sample_order = ['C', 'T', 'D']

# Step 3: Create the strip plot for individual replicates with the custom order
sns.stripplot(data=merged_data, x='age_group', y='total_signal_all_areas',
               hue='sample_group', dodge=True, marker='o', alpha=1, size=7, 
               palette='Set2', hue_order=custom_sample_order)

# Step 4: Customize the plot
plt.title('Total Signal Area by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Total Signal Area')
plt.legend(title='Sample Group', bbox_to_anchor=(1.05, 1), loc='upper left')

# Step 5: Adjust layout and display
plt.tight_layout()
plt.show()




# %%
