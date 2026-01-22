#!/usr/bin/env python
"""Plot histogram of how many timepoints each subject has."""

import pandas as pd
import matplotlib.pyplot as plt

# Read the data
df = pd.read_csv('tcr_dataset/repertoire_index.tsv', sep='\t')

# Count timepoints per subject
timepoints_per_subject = df.groupby('subject_id').size()

# Create histogram
plt.figure(figsize=(8, 6))
plt.hist(timepoints_per_subject.values, bins=range(1, timepoints_per_subject.max() + 2), 
         edgecolor='black', align='left')
plt.xlabel('Number of Timepoints')
plt.ylabel('Number of Subjects')
plt.title('Distribution of Timepoints per Subject')
plt.xticks(range(1, timepoints_per_subject.max() + 1))
plt.grid(axis='y', alpha=0.3)

# Add summary stats as text
plt.text(0.95, 0.95, f'Total subjects: {len(timepoints_per_subject)}\n'
                     f'Mean: {timepoints_per_subject.mean():.1f}\n'
                     f'Median: {timepoints_per_subject.median():.0f}',
         transform=plt.gca().transAxes, ha='right', va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('timepoints_per_subject_histogram.png', dpi=150)
print(f"Saved plot to timepoints_per_subject_histogram.png")
print(f"\nTimepoints per subject:\n{timepoints_per_subject.sort_values(ascending=False)}")
