import torch

# Define file path
input_file = "data/spikeprot0430/virus_tokenized_data_from_gde_subsampled.pt"

# Load the subsampled data
subsampled_data = torch.load(input_file)

# Collect unique strings from the 'time-loc' key
unique_time_loc_strings = set()
if subsampled_data and isinstance(subsampled_data, list):
    for element in subsampled_data:
        if isinstance(element, dict) and 'time-loc' in element:
            unique_time_loc_strings.add(element['time-loc'])

# Print the number of unique strings
print(f"Number of unique 'time-loc' strings: {len(unique_time_loc_strings)}")
