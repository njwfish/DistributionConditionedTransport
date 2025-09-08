import torch

def print_structure(obj, indent=0, key_name=""):
    """Recursively print the structure of an object with proper indentation."""
    prefix = "  " * indent
    
    if isinstance(obj, dict):
        if key_name:
            print(f"{prefix}{key_name}: dict with {len(obj)} keys")
        for key, value in obj.items():
            print_structure(value, indent + 1, key)
    elif isinstance(obj, (list, tuple)):
        type_name = "list" if isinstance(obj, list) else "tuple"
        print(f"{prefix}{key_name}: {type_name} with length {len(obj)}")
        # Optionally print structure of first element if it exists and is complex
        if len(obj) > 0 and isinstance(obj[0], (dict, list, tuple)):
            print(f"{prefix}  First element structure:")
            print_structure(obj[0], indent + 2, "")
    else:
        try:
            length = len(obj)
            print(f"{prefix}{key_name}: {type(obj).__name__} with length {length}")
        except TypeError:
            # In case the value doesn't have a length (e.g., single number, scalar)
            print(f"{prefix}{key_name}: {type(obj).__name__} (no length) = {obj}")

# Load the file
data = torch.load('data/spikeprot0430/virus_tokenized_data_for_tde_downsampled100_filtered_test.pt')

print(f"Total number of dictionaries: {len(data)}\n")

# Iterate through each dictionary in the list
for i, dictionary in enumerate(data):
    print(f"Dictionary {i}:")
    print_structure(dictionary, 1)
    print()
