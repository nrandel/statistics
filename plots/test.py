# %%
import pandas as pd

# Load the CSV file
file_path = "/Users/nadine/Documents/paper/Naomi-NS-maturation/R_tag-analysis/data/cable_lengths-IN_mature_soma-golgi.csv"  # Change this to your actual file path
df = pd.read_csv(file_path)

# Count occurrences of each skid
skid_counts = df["skid"].value_counts()

# Filter skids that appear more than once
duplicates = skid_counts[skid_counts > 1]

# Print results
if duplicates.empty:
    print("No duplicate skids found.")
else:
    print("Duplicate skids and their counts:\n", duplicates)
# %%
