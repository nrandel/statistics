#%%
""""For a single text file - compare first and last frame"""
#%%
import pandas as pd

file_path = r'/Users/nadine/Desktop/Tubingen/TRP_temp/Temperature/temp-4d_2/untitled folder/4d-18c.zvi  Ch0_path_length.txt'
df = pd.read_csv(file_path, skiprows=[1], delimiter='\t', engine='python')

# Show a preview
print(df.head())
#%%
# Optional: Reshape from wide to long format (one row per frame-particle pair)
data_long = pd.DataFrame()

for i in range(1, 76):
    data_long = pd.concat([data_long, pd.DataFrame({
        'Frame': df['Frame'],
        'Particle': i,
        'X': df[f'X{i}'],
        'Y': df[f'Y{i}'],
        'Flag': df[f'Flag{i}']
    })], ignore_index=True)

# Display reshaped data
print(data_long.head())

#%%
# First, clean the 'Frame' column to extract numeric frames only
data_long['Frame_clean'] = pd.to_numeric(data_long['Frame'].str.extract(r'(\d+)')[0], errors='coerce')

# Drop rows where frame couldn't be converted
data_long = data_long.dropna(subset=['Frame_clean'])
data_long['Frame_clean'] = data_long['Frame_clean'].astype(int)


#%%
def analyze_displacement(data_long, frame_start=3, frame_end=70):
    pos_start = data_long[data_long['Frame_clean'] == frame_start][['Particle', 'X']]
    pos_end = data_long[data_long['Frame_clean'] == frame_end][['Particle', 'X']]

    print(f"Found {len(pos_start)} entries for frame {frame_start}")
    print(f"Found {len(pos_end)} entries for frame {frame_end}")

    merged = pd.merge(pos_start, pos_end, on='Particle', suffixes=('_start', '_end'))

    # Convert X columns to numeric
    merged['X_start'] = pd.to_numeric(merged['X_start'], errors='coerce')
    merged['X_end'] = pd.to_numeric(merged['X_end'], errors='coerce')

    # Drop rows with missing data
    merged = merged.dropna(subset=['X_start', 'X_end'])

    print(f"Merged entries (after cleaning): {len(merged)}")

    merged['Displacement'] = merged['X_end'] - merged['X_start']

    mean_disp = merged['Displacement'].mean()
    std_disp = merged['Displacement'].std()

    print(f"Displacement from frame {frame_start} to {frame_end}:")
    print(f"Mean: {mean_disp:.2f}, Std Dev: {std_disp:.2f}")
    
    return merged[['Particle', 'Displacement']]


# %%
analyze_displacement(data_long)

# %%
""""For a multiple text file in a directory - compare first and last frame"""
import os
import pandas as pd

# Path to the directory containing the text files
folder_path = r'/Users/nadine/Desktop/Tubingen/TRP_temp/Temperature/temp-4d_2/untitled folder'

def analyze_displacement(data_long, frame_start=3, frame_end=None):
    # Convert 'Frame' to numeric values to avoid type issues
    data_long['Frame'] = pd.to_numeric(data_long['Frame'], errors='coerce')

    # If frame_end is not provided, set it to the last frame minus 3
    if frame_end is None:
        frame_end = data_long['Frame'].max() - 3

    # Filter data for the specified frames
    frame_start_data = data_long[data_long['Frame'] == frame_start]
    frame_end_data = data_long[data_long['Frame'] == frame_end]

    # Merge the data based on particle ID
    merged = pd.merge(frame_start_data, frame_end_data, on='Particle', suffixes=('_start', '_end'))

    # Ensure that X_start and X_end are numeric
    merged['X_start'] = pd.to_numeric(merged['X_start'], errors='coerce')
    merged['X_end'] = pd.to_numeric(merged['X_end'], errors='coerce')

    # Print column names to debug
    print("Merged DataFrame columns:", merged.columns)

    # Calculate displacement (positive and negative values)
    merged['Displacement'] = merged['X_end'] - merged['X_start']
    
    # Check if 'Age_start' and 'Temperature_start' columns exist in the merged dataframe
    if 'Age_start' in merged.columns and 'Temperature_start' in merged.columns:
        # Group by Age and Temperature to get the mean and standard deviation
        grouped = merged.groupby(['Age_start', 'Temperature_start'])['Displacement'].agg(['mean', 'std']).reset_index()

        # Output the grouped results
        print("Average and Standard Deviation of Displacement per Age and Temperature:")
        print(grouped)

        return grouped
    else:
        print("Error: 'Age_start' or 'Temperature_start' column is missing from merged DataFrame.")
        return None

def process_files(folder_path):
    all_data = pd.DataFrame()  # Initialize an empty dataframe to hold all the data
    
    # Loop through all the files in the directory
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            
            # Extract age and temperature from the filename (e.g., '4d-10c')
            name_parts = file_name.split(' ')
            age_temp = name_parts[0].replace('.zvi', '')  # Remove '.zvi' part
            age, temp = age_temp.split('-')  # Split '4d' and '10c'
            
            # Read the data from the text file
            df = pd.read_csv(file_path, skiprows=[1], delimiter='\t', engine='python')

            # Initialize data_long
            data_long = pd.DataFrame()

            for i in range(1, 76):
                # Check if the necessary columns exist before trying to access them
                if f'X{i}' in df.columns and f'Y{i}' in df.columns and f'Flag{i}' in df.columns:
                    data_long = pd.concat([data_long, pd.DataFrame({
                        'Frame': df['Frame'],
                        'Particle': i,
                        'X': df[f'X{i}'],
                        'Y': df[f'Y{i}'],
                        'Flag': df[f'Flag{i}'],
                        'Age': age,  # Extracted from the filename
                        'Temperature': temp  # Extracted from the filename
                    })], ignore_index=True)
                else:
                    print(f"Warning: Columns X{i}, Y{i}, or Flag{i} not found in file: {file_name}")
            
            # Combine the reshaped data with the all_data dataframe
            all_data = pd.concat([all_data, data_long], ignore_index=True)
    
    return all_data

# Process all files in the folder
data_long = process_files(folder_path)

# Analyze displacement for the entire dataset
grouped = analyze_displacement(data_long)

# Optionally, you can save the analysis results
# grouped.to_csv('/path/to/save/grouped_displacement_data.csv', index=False)


# %%

"""TEST for multiple files adjusted after perl script GJ"""


import os
import pandas as pd

# Path to the directory containing the text files
folder_path = r'/Users/nadine/Desktop/Tubingen/TRP_temp/Temperature/temp-4d_2/untitled folder'

def analyze_displacement(data_long, frame_start=3, frame_end=None):
    # Convert 'Frame' to numeric values to avoid type issues
    data_long['Frame'] = pd.to_numeric(data_long['Frame'], errors='coerce')

    # If frame_end is not provided, set it to the last frame minus 3
    if frame_end is None:
        frame_end = data_long['Frame'].max() - 3

    # Filter data for the specified frames
    frame_start_data = data_long[data_long['Frame'] == frame_start]
    frame_end_data = data_long[data_long['Frame'] == frame_end]

    # Merge the data based on particle ID
    merged = pd.merge(frame_start_data, frame_end_data, on='Particle', suffixes=('_start', '_end'))

    # Ensure that X_start and X_end are numeric
    merged['X_start'] = pd.to_numeric(merged['X_start'], errors='coerce')
    merged['X_end'] = pd.to_numeric(merged['X_end'], errors='coerce')

    # Print column names to debug
    print("Merged DataFrame columns:", merged.columns)

    # Calculate displacement (positive and negative values)
    merged['Displacement'] = merged['X_end'] - merged['X_start']
    
    # Check if 'Age_start' and 'Temperature_start' columns exist in the merged dataframe
    if 'Age_start' in merged.columns and 'Temperature_start' in merged.columns:
        # Group by Age and Temperature to get the mean and standard deviation
        grouped = merged.groupby(['Age_start', 'Temperature_start'])['Displacement'].agg(['mean', 'std']).reset_index()

        # Output the grouped results
        print("Average and Standard Deviation of Displacement per Age and Temperature:")
        print(grouped)

        return grouped
    else:
        print("Error: 'Age_start' or 'Temperature_start' column is missing from merged DataFrame.")
        return None

def process_files(folder_path):
    all_data = pd.DataFrame()  # Initialize an empty dataframe to hold all the data
    
    # Loop through all the files in the directory
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            
            # Extract age and temperature from the filename (e.g., '4d-10c')
            name_parts = file_name.split(' ')
            age_temp = name_parts[0].replace('.zvi', '')  # Remove '.zvi' part
            age, temp = age_temp.split('-')  # Split '4d' and '10c'
            
            # Read the data from the text file
            df = pd.read_csv(file_path, skiprows=[1], delimiter='\t', engine='python')

            # Initialize data_long
            data_long = pd.DataFrame()

            for i in range(1, 76):  # Assuming 75 particles, as per the previous pattern
                # Check if the necessary columns exist before trying to access them
                if f'X{i}' in df.columns and f'Y{i}' in df.columns and f'Flag{i}' in df.columns:
                    data_long = pd.concat([data_long, pd.DataFrame({
                        'Frame': df['Frame'],
                        'Particle': i,
                        'X': df[f'X{i}'],
                        'Y': df[f'Y{i}'],
                        'Flag': df[f'Flag{i}'],
                        'Age': age,  # Extracted from the filename
                        'Temperature': temp  # Extracted from the filename
                    })], ignore_index=True)
                else:
                    print(f"Warning: Columns X{i}, Y{i}, or Flag{i} not found in file: {file_name}")
            
            # Combine the reshaped data with the all_data dataframe
            all_data = pd.concat([all_data, data_long], ignore_index=True)
    
    return all_data

# Process all files in the folder
data_long = process_files(folder_path)

# Analyze displacement for the entire dataset
grouped = analyze_displacement(data_long)

# Optionally, you can save the analysis results
# grouped.to_csv('/path/to/save/grouped_displacement_data.csv', index=False)

# %%
"""Plotting of horizontal displacement after GJ"""
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Use TkAgg backend (if necessary)
matplotlib.use('TkAgg')

# Path to the directory containing the text files
folder_path = r'/Users/nadine/Desktop/Tubingen/TRP_temp/Temperature/temp-4d_2/untitled folder'

def analyze_displacement(data_long, frame_start=3, frame_end=None):
    # Convert 'Frame' to numeric values to avoid type issues
    data_long['Frame'] = pd.to_numeric(data_long['Frame'], errors='coerce')

    # If frame_end is not provided, set it to the last frame minus 3
    if frame_end is None:
        frame_end = data_long['Frame'].max() - 3

    # Filter data for the specified frames
    frame_start_data = data_long[data_long['Frame'] == frame_start]
    frame_end_data = data_long[data_long['Frame'] == frame_end]

    # Merge the data based on particle ID
    merged = pd.merge(frame_start_data, frame_end_data, on='Particle', suffixes=('_start', '_end'))

    # Ensure that X_start and X_end are numeric
    merged['X_start'] = pd.to_numeric(merged['X_start'], errors='coerce')
    merged['X_end'] = pd.to_numeric(merged['X_end'], errors='coerce')

    # Calculate displacement (positive and negative values)
    merged['Displacement'] = merged['X_end'] - merged['X_start']
    
    # Check if 'Age_start' and 'Temperature_start' columns exist in the merged dataframe
    if 'Age_start' in merged.columns and 'Temperature_start' in merged.columns:
        # Group by Age and Temperature to get the mean and standard deviation
        grouped = merged.groupby(['Age_start', 'Temperature_start'])['Displacement'].agg(['mean', 'std']).reset_index()

        return grouped
    else:
        print("Error: 'Age_start' or 'Temperature_start' column is missing from merged DataFrame.")
        return None

def process_files(folder_path):
    all_data = pd.DataFrame()  # Initialize an empty dataframe to hold all the data
    
    # Loop through all the files in the directory
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            
            # Extract age and temperature from the filename (e.g., '4d-10c')
            name_parts = file_name.split(' ')
            age_temp = name_parts[0].replace('.zvi', '')  # Remove '.zvi' part
            age, temp = age_temp.split('-')  # Split '4d' and '10c'
            
            # Read the data from the text file
            df = pd.read_csv(file_path, skiprows=[1], delimiter='\t', engine='python')

            # Initialize data_long
            data_long = pd.DataFrame()

            for i in range(1, 76):  # Assuming 75 particles, as per the previous pattern
                # Check if the necessary columns exist before trying to access them
                if f'X{i}' in df.columns and f'Y{i}' in df.columns and f'Flag{i}' in df.columns:
                    data_long = pd.concat([data_long, pd.DataFrame({
                        'Frame': df['Frame'],
                        'Particle': i,
                        'X': df[f'X{i}'],
                        'Y': df[f'Y{i}'],
                        'Flag': df[f'Flag{i}'],
                        'Age': age,  # Extracted from the filename
                        'Temperature': temp  # Extracted from the filename
                    })], ignore_index=True)
                else:
                    print(f"Warning: Columns X{i}, Y{i}, or Flag{i} not found in file: {file_name}")
            
            # Combine the reshaped data with the all_data dataframe
            all_data = pd.concat([all_data, data_long], ignore_index=True)
    
    return all_data

# Process all files in the folder
data_long = process_files(folder_path)

# Analyze displacement for the entire dataset
grouped = analyze_displacement(data_long)

# Plotting the results using seaborn (Barplot)
if grouped is not None:
    # Create a barplot for the mean displacement
    plt.figure(figsize=(10, 6))
    sns.barplot(data=grouped, x='Age_start', y='mean', hue='Temperature_start', errorbar=None)
    
    # Adding labels and title
    plt.title('Horizontal Displacement by Age and Temperature')
    plt.xlabel('Age')
    plt.ylabel('Mean Displacement (X)')
    plt.legend(title='Temperature', loc='best')
    
    # Save the plot as a PNG file
    plt.savefig('/Users/nadine/Desktop/untitled/plot.png', dpi=300)  # Adjust the path as needed
    
    # Close the plot to free memory
    plt.close()

else:
    print("No data to plot.")



# %%
