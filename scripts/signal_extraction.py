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

# Loop through each sample folder in the main directory
for sample_folder in os.listdir(main_directory):
    sample_path = os.path.join(main_directory, sample_folder)
    
    # Only proceed if it's a directory
    if os.path.isdir(sample_path):
        # Find all CSV files within the sample folder (and subfolders, if needed)
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

        # Combine all DataFrames for this sample
        combined_df = pd.concat(dfs, ignore_index=True)

        # Step 1: Add a new column for signal area for each slice (Signal Area = %Area * Area / 100)
        combined_df['Signal_Area'] = (combined_df['%Area'] / 100) * combined_df['Area']

        # Step 2: Group by filename and calculate total area, total signal area, and weighted average %Area per region
        sample_results = combined_df.groupby('filename').agg(
            total_area=('Area', 'sum'),                      # Sum of Area for each 3D stack (filename)
            total_signal_area=('Signal_Area', 'sum'),        # Sum of Signal Area for each stack
            weighted_avg_percent_area=('Signal_Area', lambda x: x.sum() / combined_df.loc[x.index, 'Area'].sum() * 100)  # Weighted avg %Area
        ).reset_index()

        # Add the sample identifier to the results
        sample_results['sample'] = sample_folder
        
        # Append the sample's results to the main list
        all_results.append(sample_results)

# Step 3: Combine results for all samples
final_results = pd.concat(all_results, ignore_index=True)

# Display the final result
print(final_results[['sample', 'filename', 'total_area', 'total_signal_area', 'weighted_avg_percent_area']])


# %%
