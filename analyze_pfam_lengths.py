import gzip
import numpy as np

data_dir = 'data/pfam'
fasta_file = f'{data_dir}/Pfam-A.fasta.gz'
lines_to_read = 10**8

print(f'Reading {fasta_file}...')

sequence_lengths = []
current_seq = ""

with gzip.open(fasta_file, 'rt') as f:
    for i, line in enumerate(f):
        if i >= lines_to_read:
            break
        
        if line.startswith('>'):
            # Save the previous sequence if it exists
            if current_seq:
                sequence_lengths.append(len(current_seq))
            current_seq = ""
        else:
            current_seq += line.strip()
        
        if (i + 1) % 1_000_000 == 0:
            print(f'Processed {i + 1:,} lines, found {len(sequence_lengths):,} sequences so far...')

# Don't forget the last sequence
if current_seq:
    sequence_lengths.append(len(current_seq))

print(f'\n--- Results ---')
print(f'Total sequences found: {len(sequence_lengths):,}')
print(f'Shortest sequence length: {min(sequence_lengths):,}')
print(f'Longest sequence length: {max(sequence_lengths):,}')
print(f'Mean sequence length: {np.mean(sequence_lengths):.2f}')
print(f'Median sequence length: {np.median(sequence_lengths):.2f}')
