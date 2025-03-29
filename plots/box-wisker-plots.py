# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv

# %%
# Import csv as panda dataframe

#EdU exp
#df = pd.read_csv("/Users/nadine/Documents/paper/Naomi-NS-maturation/BF_EdU-larvae/BF_EdU-larvae_size-merge_um.csv") 

# Feeding exp
df = pd.read_csv("/Users/nadine/Documents/paper/Naomi-NS-maturation/Feeding-Experiments-AxioZoom_Exeter_1_7-13-2023/Feeding-size_um.csv")  


# %%
# box and whisker plot with label on left side
#figure(figsize=(10, 8), dpi=100)

ax = df.boxplot(vert=False, grid = False, color=dict(boxes='k', whiskers='k', medians='b', caps='k'), showfliers=False) #showfliers=False hide outlier datapoints

# %%
# Export plot as svg

ax.figure.savefig('/Users/nadine/Documents/paper/Naomi-NS-maturation/generated_plots/feeding_experiment_axiozoom_um.svg')
# %%
# test
# line plot with media

# Drop the first column
data = df.iloc[:, 1:]

# Drop NaN values per column
filtered_data = np.nanmedian(data.to_numpy(), axis=0)


# Calculate median
median = np.nanmedian(filtered_data)

# Create the line plot
plt.figure(figsize=(10, 6))
plt.plot(filtered_data, label='Data')
plt.axhline(y=median, color='r', linestyle='--', label='Median')
plt.title('Line Plot with Median')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# %%
# stacked bar plot

# Import csv
df = pd.read_csv("/Users/nadine/Documents/paper/Naomi-NS-maturation/Feeding-Experiments-AxioZoom_Exeter_1_7-13-2023/gut_content_T.csv")  

# Plot stacked bar chart
plt.figure(figsize=(10, 6))
df.plot(kind='bar', stacked=True, width=0.8)

plt.title('Number of Treatments for Each Condition and Treatment')
plt.xlabel('Treatment')
plt.ylabel('Number of Treatments')
plt.xticks(rotation=45)
#plt.legend(title='Condition')
#plt.grid(True)
#plt.tight_layout()

#plt.show()


# Save plot as SVG
plt.savefig('/Users/nadine/Documents/paper/Naomi-NS-maturation/generated_plots/gut_content.svg', format='svg')


# %%
# test combined stacked bar (gut) and line plot (yolk)
import pandas as pd
import matplotlib.pyplot as plt

# Load data for stacked bar chart
df_gut = pd.read_csv("/Users/nadine/Documents/paper/Naomi-NS-maturation/summary-gut-yolk-csv/tetraselmis_gut.csv")

# Load data for line plot
data_yolk = pd.read_csv("/Users/nadine/Documents/paper/Naomi-NS-maturation/summary-gut-yolk-csv/yolk-tetraselmis.csv")

# Calculate the total number of features per condition for the line plot data
data_yolk['Total_Features'] = data_yolk['no yolk'] + data_yolk['yolk']

# Calculate the percentage of each feature relative to the total number of features
data_yolk['no yolk_Percentage'] = (data_yolk['no yolk'] / data_yolk['Total_Features']) * 100
data_yolk['yolk_Percentage'] = (data_yolk['yolk'] / data_yolk['Total_Features']) * 100

# Plot combined plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot stacked bar chart on primary y-axis
df_gut.plot(kind='bar', stacked=True, width=0.4, ax=ax1, color=['#1f77b4', '#ff7f0e'])

# Set labels and title
ax1.set_title('Relative Amount of Features per Condition')
ax1.set_xlabel('Condition')
ax1.set_ylabel('Count')

# Create a secondary y-axis for the line plot
ax2 = ax1.twinx()

# Plot line plot on secondary y-axis
ax2.plot(data_yolk.index, data_yolk['yolk_Percentage'], marker='o', label='yolk', color='black', linewidth=4)

# Set label for secondary y-axis
ax2.set_ylabel('Percentage')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Display legends
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines2, labels2, loc='best')

# Save plot as SVG
plt.savefig('/Users/nadine/Desktop/gut-yolk-Gmaria.svg', format='svg')


#plt.show()



# %%
# test yolk to percent and line plot


# Load the dataset
data = pd.read_csv("/Users/nadine/Documents/paper/Naomi-NS-maturation/summary-gut-yolk-csv/yolk-Gmarina.csv")  

# Assuming the dataset has 'Condition' as the index and 'Feature_Absent', 'Feature_Not_Absent' as columns

# Calculate the total number of features per condition
data['Total_Features'] = data['no yolk'] + data['yolk']

# Calculate the percentage of each feature relative to the total number of features
data['yolk_Percentage'] = (data['yolk'] / data['Total_Features']) * 100

# Plot line plot
plt.figure(figsize=(10, 6))

# Plot the relative percentage of Feature_Not_Absent
plt.plot(data.index, data['yolk_Percentage'], marker='o', label='yolk')

plt.title('Relative Amount of Features per Condition')
plt.xlabel('Condition')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend()

plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt

# List of file paths for CSV files
csv_files = [
    "/Users/nadine/Documents/paper/Naomi-NS-maturation/summary-gut-yolk-csv/tetraselmis_gut.csv",
    "/Users/nadine/Documents/paper/Naomi-NS-maturation/summary-gut-yolk-csv/Gmarina_gut.csv",

]

# Create a list to store DataFrames for each CSV file
dfs = []

# Load and store data from each CSV file
for file_path in csv_files:
    df = pd.read_csv(file_path)
    dfs.append(df)

# Extract column names
column_names = dfs[0].columns

# Plot stacked bar plots for each CSV file side by side in the same plot
plt.figure(figsize=(10, 6))

# Set initial position for the bars
bar_width = 0.25
num_files = len(dfs)
num_columns = len(column_names)
bar_positions = np.arange(num_columns)

# Calculate the gap between the bars
gap = bar_width / (num_files + 1)

# Plot stacked bar plots for each CSV file
for i, df in enumerate(dfs):
    # Adjust the position for each stacked bar plot
    adjusted_bar_positions = [pos + i * (bar_width + gap) for pos in bar_positions]

    # Plot stacked bar plots for each column in the DataFrame
    for j, col in enumerate(column_names):
        # Extract data for the column
        data = df[col]
        
        # Plot stacked bar plot (gap between the bars by dividing the total width of the bars by the number of files plus 1 (for the gaps) and adjusting the positions accordingly)
        plt.bar(adjusted_bar_positions[j], data, width=bar_width, label=f'CSV File {i+1} - {col}')

# Set labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Stacked Bar Plots for Multiple CSV Files')
plt.xticks([pos + (num_files - 1) * (bar_width + gap) / 2 for pos in bar_positions], column_names, rotation=45)
plt.legend()

# Show plot
plt.tight_layout()
plt.show()




# %%
# Plot number of neurons (mature/ immature)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV (assuming no header and a single row of data)
csv_file = "/Users/nadine/Documents/paper/Naomi-NS-maturation/neuron-catmaid-output03-2025.csv"  
df = pd.read_csv(csv_file)

# Transpose the DataFrame to switch rows and columns
df = df.transpose()

# Set the column names as headers for better readability
df.columns = ['Value']

# Plot the bar chart
df.plot(kind='bar', legend=False)

# Add title and labels
plt.title('Bar Plot Example')
plt.xlabel('Category')
plt.ylabel('Value')

# Set the x-axis labels to the actual column names
plt.xticks(range(len(df.index)), df.index, rotation=45)  # Use index for categories

# Show the plot
plt.show()


# %%
# Stacked plot number of neurons (mature/ immature)

import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV data with headers
csv_file = "/Users/nadine/Documents/paper/Naomi-NS-maturation/neuron-catmaid-output03-2025.csv"  
df = pd.read_csv(csv_file)

# Transpose the DataFrame to switch rows and columns
df = df.transpose()

# Set the correct column name for the data
df.columns = ['Value']

# Reset the index to retain the original column names (so we can extract prefixes)
df.reset_index(inplace=True)

# Extract the prefixes (SN, IN, MN) and maturity status (mature, immature)
df['Prefix'] = df['index'].str.extract(r'^(SN|IN|MN)')[0]
df['Maturity'] = df['index'].str.extract(r'-(mature|immature)')[0]

# Define the order of categories
category_order = ['SN', 'IN', 'MN']

# Pivot the DataFrame
pivoted_df = df.pivot_table(index='Prefix', columns='Maturity', values='Value', aggfunc='sum')

# Reorder the DataFrame based on the predefined order
pivoted_df = pivoted_df.loc[category_order]

# Set figure size
fig, ax = plt.subplots(figsize=(3, 4))  # Adjust width and height here

# Create the stacked bar plot with adjustable bar width
pivoted_df.plot(kind='bar', stacked=True, width=0.4, ax=ax)  # Adjust bar width here

# Add title and labels
plt.title('Cumulative Stacked Bar Plot by Prefix and Maturity')
plt.xlabel('Prefix')
plt.ylabel('Value')

# Save the plot as an SVG file
plt.savefig('/Users/nadine/Documents/paper/Naomi-NS-maturation/generated_plots/stacked_bar_plot.svg', format='svg')

# Show the plot
plt.show()

# %%
# Stacked plot number of neurons (mature/ immature neurons) PLUS dendrites 

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('/Users/nadine/Documents/paper/Naomi-NS-maturation/catmaid_neurons_dendrite_25-3-25.csv')

# Transpose the DataFrame so the headers become a column
df = df.transpose()

# Set column name
df.columns = ['Value']

# Reset index to extract categories
df.reset_index(inplace=True)

# Extract Prefix (SN, IN, MN)
df['Prefix'] = df['index'].str.extract(r'^(SN|IN|MN)')[0]

# Extract Maturity (mature, immature)
df['Maturity'] = df['index'].str.extract(r'-(mature|immature)')[0]

# Initialize 'Dendrite Status' with empty string
df['Dendrite Status'] = ''

# Apply dendrite extraction **only to SN**
df.loc[df['Prefix'] == 'SN', 'Dendrite Status'] = df['index'].str.extract(r'dendr (mat|immat)')[0]

# Fill NaN values for 'Dendrite Status' (only affects IN and MN)
df['Dendrite Status'] = df['Dendrite Status'].fillna('No Dendrite')

# Drop any rows with missing Prefix or Maturity (to ensure valid data)
df.dropna(subset=['Prefix', 'Maturity'], inplace=True)

# Convert Value column to numeric (handling potential string conversion issues)
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

# Debugging check
print("Processed DataFrame:\n", df)

# Define order for the x-axis categories
category_order = ['SN', 'IN', 'MN']

# Pivot table: 
pivoted_df = df.pivot_table(
    index='Prefix', 
    columns=['Maturity', 'Dendrite Status'], 
    values='Value', 
    aggfunc='sum'
)

# Ensure correct ordering
pivoted_df = pivoted_df.reindex(category_order)

# Debugging check
print("Pivoted DataFrame:\n", pivoted_df)

# Plot only if pivoted_df is not empty
if pivoted_df.empty:
    print("Error: Pivoted DataFrame is empty. Check data formatting.")
else:
    # Set figure size
    fig, ax = plt.subplots(figsize=(3, 4))  # Adjust width and height as needed

    # Create stacked bar plot
    pivoted_df.plot(kind='bar', stacked=True, width=0.4, ax=ax)

    # Add title and labels
    plt.title('Stacked Bar Plot with SN Subcategories')
    plt.xlabel('Neuron Type')
    plt.ylabel('Value')

    # Save the plot as an SVG file
    plt.savefig('/Users/nadine/Documents/paper/Naomi-NS-maturation/generated_plots/stacked_bar_plot-dendrites.svg', format='svg')

    # Show the plot
    plt.show()


# %%

# Violin plots for tag distance, grouped for neuron typres

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import re
import os

# Define the order of neuron types
neuron_order = ["SN_mature", "SN_immature", "IN_mature", "IN_immature", "MN_mature", "MN_immature"]

# Find all relevant CSV files
csv_files = glob.glob("/Users/nadine/Documents/paper/Naomi-NS-maturation/R_tag-analysis/data/cable_lengths-*.csv")

# Dictionary to store data by tag-pair
data_dict = {}

# Regular expression to extract neuron type and tag pair
pattern = re.compile(r"cable_lengths-(\w+)_(\w+-\w+)\.csv")

for file in csv_files:
    match = pattern.search(file)
    if match:
        neuron_type, tag_pair = match.groups()
        
        # Read CSV file
        df = pd.read_csv(file)
        
        # Print column names for debugging
        print(f"Columns in {file}: {df.columns.tolist()}")

        # Ensure column name consistency
        df.columns = df.columns.str.strip().str.replace(".", "_", regex=True)

        # Print column names after fixing
        print(f"Fixed columns in {file}: {df.columns.tolist()}")

        df.columns = df.columns.str.strip()  # Remove any hidden spaces
        print(f"Cleaned column names: {df.columns.tolist()}")

        # Remove NA values
        df = df.dropna(subset=["cable_length"])
        
        # Add neuron type column
        df["neuron_type"] = neuron_type
        
        # Store data
        if tag_pair not in data_dict:
            data_dict[tag_pair] = []
        data_dict[tag_pair].append(df)

# Define the directory to save plots
save_directory = "/Users/nadine/Documents/paper/Naomi-NS-maturation/R_tag-analysis/plots"

# Plot violin plots for each tag-pair
for tag_pair, df_list in data_dict.items():
    combined_df = pd.concat(df_list)
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=combined_df, x="neuron_type", y="cable_length", order=neuron_order)
    
    plt.title(f"Cable Length Distribution for {tag_pair}")
    plt.xlabel("Neuron Type")
    plt.ylabel("Cable Length")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot as SVG
    save_path = os.path.join(save_directory, f"violin_{tag_pair}.svg")
    plt.savefig(save_path, format="svg")  # Save as SVG format
    
    plt.show()



# %%

# Violin plots for tag distance, grouped for neuron typres and adjusted violin plot by normalizing width

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import re
import os

# Define the order of neuron types
neuron_order = ["SN_mature", "SN_immature", "IN_mature", "IN_immature", "MN_mature", "MN_immature"]

# Find all relevant CSV files
csv_files = glob.glob("/Users/nadine/Documents/paper/Naomi-NS-maturation/R_tag-analysis/data/cable_lengths-*.csv")

# Dictionary to store data by tag-pair
data_dict = {}

# Regular expression to extract neuron type and tag pair
pattern = re.compile(r"cable_lengths-(\w+)_(\w+-\w+)\.csv")

for file in csv_files:
    match = pattern.search(file)
    if match:
        neuron_type, tag_pair = match.groups()
        
        # Read CSV file
        df = pd.read_csv(file)
        
        # Print column names for debugging
        print(f"Columns in {file}: {df.columns.tolist()}")

        # Ensure column name consistency
        df.columns = df.columns.str.strip().str.replace(".", "_", regex=True)

        # Print column names after fixing
        print(f"Fixed columns in {file}: {df.columns.tolist()}")

        df.columns = df.columns.str.strip()  # Remove any hidden spaces
        print(f"Cleaned column names: {df.columns.tolist()}")

        # Remove NA values
        df = df.dropna(subset=["cable_length"])
        
        # Add neuron type column
        df["neuron_type"] = neuron_type
        
        # Store data
        if tag_pair not in data_dict:
            data_dict[tag_pair] = []
        data_dict[tag_pair].append(df)

# Define the directory to save plots
save_directory = "/Users/nadine/Documents/paper/Naomi-NS-maturation/R_tag-analysis/plots"

# Plot violin plots for each tag-pair
for tag_pair, df_list in data_dict.items():
    combined_df = pd.concat(df_list)
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=combined_df, x="neuron_type", y="cable_length", order=neuron_order, scale="width")
    
    plt.title(f"Cable Length Distribution for {tag_pair}")
    plt.xlabel("Neuron Type")
    plt.ylabel("Cable Length")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot as SVG
    save_path = os.path.join(save_directory, f"violin_{tag_pair}.svg")
    plt.savefig(save_path, format="svg")  # Save as SVG format
    
    plt.show()
# %%
"""used for paper"""
# Violin plots for tag distance, grouped for neuron typres and add annotations

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import re
import os

# Define the order of neuron types
neuron_order = ["SN_mature", "SN_immature", "IN_mature", "IN_immature", "MN_mature", "MN_immature"]

# Find all relevant CSV files
csv_files = glob.glob("/Users/nadine/Documents/paper/Naomi-NS-maturation/R_tag-analysis/data/cable_lengths-*.csv")

# Dictionary to store data by tag-pair
data_dict = {}

# Regular expression to extract neuron type and tag pair
pattern = re.compile(r"cable_lengths-(\w+)_(\w+-\w+)\.csv")

for file in csv_files:
    match = pattern.search(file)
    if match:
        neuron_type, tag_pair = match.groups()
        
        # Read CSV file
        df = pd.read_csv(file)
        
        # Print column names for debugging
        print(f"Columns in {file}: {df.columns.tolist()}")

        # Ensure column name consistency
        df.columns = df.columns.str.strip().str.replace(".", "_", regex=True)

        # Print column names after fixing
        print(f"Fixed columns in {file}: {df.columns.tolist()}")

        df.columns = df.columns.str.strip()  # Remove any hidden spaces
        print(f"Cleaned column names: {df.columns.tolist()}")

        # Remove NA values
        df = df.dropna(subset=["cable_length"])
        
        # Add neuron type column
        df["neuron_type"] = neuron_type
        
        # Store data
        if tag_pair not in data_dict:
            data_dict[tag_pair] = []
        data_dict[tag_pair].append(df)

# Define the directory to save plots
save_directory = "/Users/nadine/Documents/paper/Naomi-NS-maturation/R_tag-analysis/plots"

# Plot violin plots with annotations for the number of samples
for tag_pair, df_list in data_dict.items():
    combined_df = pd.concat(df_list)
    
    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(data=combined_df, x="neuron_type", y="cable_length", order=neuron_order)
    
    # Add annotations for the number of samples
    for i, neuron_type in enumerate(neuron_order):
        count = combined_df[combined_df["neuron_type"] == neuron_type].shape[0]
        ax.text(i, 0.9, f'n = {count}', horizontalalignment='center', size=12, color='black', weight='semibold')
    
    plt.title(f"Cable Length Distribution for {tag_pair}")
    plt.xlabel("Neuron Type")
    plt.ylabel("Cable Length")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot as SVG
    save_path = os.path.join(save_directory, f"violin_{tag_pair}.svg")
    plt.savefig(save_path, format="svg")  # Save as SVG format
    
    plt.show()
# %%
