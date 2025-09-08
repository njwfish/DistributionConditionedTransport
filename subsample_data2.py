import torch
import random
import numpy as np

# Define file paths
input_file = "data/spikeprot0430/tokenized_chunks/virus_tokenized_data_33_40.pt.part_2"

# Load the data
data = torch.load(input_file)

def count_x_token(seq):
    return seq.count('X')

# Print keys of the first dictionary
if data and isinstance(data, list) and isinstance(data[0], dict):
    print("Keys of the first dictionary:", data[0].keys())
else:
    print("Data is not in the expected format (list of dictionaries).")

all_seqs = []
all_x_tokens = []
all_lengths = []
seqs_with_x = 0

for d in data:
    print(d['esm_input_ids'.shape])
    all_seqs.extend(d['raw_texts'])
    all_x_tokens.extend([count_x_token(seq)/len(seq) for seq in d['raw_texts']])
    all_lengths.extend([len(seq) for seq in d['raw_texts']])
    seqs_with_x += np.sum(np.array([count_x_token(seq) for seq in d['raw_texts']]) > 0)


    
all_x_tokens = np.array(all_x_tokens)
all_lengths = np.array(all_lengths)

print(np.mean(all_lengths),np.std(all_lengths))
print(np.mean(all_x_tokens))
print(len(all_x_tokens)-np.sum(all_x_tokens > 0))
print(len(all_seqs))
print(seqs_with_x)

