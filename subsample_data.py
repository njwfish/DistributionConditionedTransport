import torch
import random

# Define file paths
input_file = "data/spikeprot0430/virus_tokenized_data_from_gde.pt"
output_file = "data/spikeprot0430/virus_tokenized_data_from_gde_subsampled.pt"

# Load the data
data = torch.load(input_file)

# Print keys of the first dictionary
if data and isinstance(data, list) and isinstance(data[0], dict):
    print("Keys of the first dictionary:", data[0].keys())
else:
    print("Data is not in the expected format (list of dictionaries).")

# Subsample 100 elements
if len(data) > 100:
    subsampled_data = random.sample(data, 100)
else:
    subsampled_data = data
    print("Dataset has less than 100 elements, using full dataset.")

# Save the subsampled data
torch.save(subsampled_data, output_file)

print(f"Subsampled data saved to {output_file}")
