import pandas as pd
import plmfit.shared_utils.utils as utils
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

# Define the wildtype sequence
wildtype = "WGGDGFYAMD"

# Define the mutation data as a dictionary where the key is the amino acid and the value is the list of binding scores
mutations = {
    'A': [0, 1, 1, 1, -1, 0, 0, 1, 0, 1],
    'V': [0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
    'L': [0, 1, 0, 1, 0, -1, 0, 1, -1, 1],
    'I': [0, 1, 1, 1, 0, 0, 0, -1, 0, -1],
    'M': [0, 1, 1, 1, 0, 1, 0, 1, 1, 1],
    'F': [1, 1, 1, 1, 0, 1, 0, 1, 1, 0],
    'Y': [1, 1, 1, 0, 0, 0, 1, 1, 1, -1],
    'W': [1, 1, 1, 0, 0, 1, 0, 0, 1, 0],
    'R': [0, 1, 1, -1, 0, 0, 0, 0, 0, -1],
    'H': [0, 1, 1, 0, -1, 0, -1, 1, 1, -1],
    'K': [0, 1, 1, -1, 0, 0, 0, 0, 0, 0],
    'D': [0, 0, 0, 1, 0, 0, -1, -1, 0, 1],
    'E': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'S': [0, 1, 1, 1, 1, 0, 0, 1, 0, 1],
    'T': [0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
    'N': [0, 1, 1, 1, 0, 0, -1, -1, 1, 0],
    'Q': [0, 1, -1, 1, 0, 0, 0, 1, 1, 0],
    'G': [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    'C': [0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
    'P': [0, 1, 0, 1, 0, 0, 0, 0, 0, 1]
}

# function to compare two strings and return the number of differences
def compare_strings(s1, s2):
    return sum([1 for i in range(len(s1)) if s1[i] != s2[i]])

# List to store each row of the final CSV file
data = []

# Iterate over each position in the wildtype sequence
for pos in range(len(wildtype)):
    # Iterate over each possible mutation at this position
    for aa, scores in mutations.items():
        score = scores[pos]
        # Check if the score is -1 (ignore these cases)
        if score != -1:
            # Create the mutated sequence
            mutated_seq = list(wildtype)
            mutated_seq[pos] = aa
            mutated_seq = ''.join(mutated_seq)
            # Append the mutated sequence and its score to the data list
            data.append([mutated_seq, score])

# Create a DataFrame from the data list
dms = pd.DataFrame(data, columns=["AASeq", "AgClass"])


# File paths for the dataset and FASTA file
csv_path_positive = "mHER_H3_AgPos.csv"
csv_path_negative = "mHER_H3_AgNeg.csv"
wildtype_path = "wild_type.json"

wildtype = json.load(open(os.path.join(script_dir, wildtype_path)))
wildtype = wildtype["wild_type"]

mutated_region_start = 98
mutated_region_end = 108
mutated_region = wildtype[mutated_region_start:mutated_region_end]
print(mutated_region)

# Load dataset from CSV file
pos_data = pd.read_csv(
    os.path.join(script_dir, csv_path_positive)
) 

neg_data = pd.read_csv(
    os.path.join(script_dir, csv_path_negative)
)
print(dms["AgClass"].value_counts())
print(pos_data['AgClass'].value_counts())
print(neg_data['AgClass'].value_counts())
data = pd.concat([dms, pos_data, neg_data])
data.drop_duplicates(subset="AASeq", keep="first", inplace=True)

data.rename(columns={"AgClass": "binary_score"}, inplace=True)

# Reset index to avoid indexing issues
data = data.reset_index(drop=True)


# calculate no_mut column based on mutated region for each chain, no mask is used
data['no_mut'] = data['AASeq'].apply(lambda x: compare_strings(mutated_region, x))

data['one_vs_rest'] = np.where(data['no_mut'] <= 1, 'train', 'test')

# Split features and target
X = data.drop('binary_score', axis=1)
y = data['binary_score']


# Separate True and False class indices
true_indices = data[data['binary_score'] == 1].index
false_indices = data[data['binary_score'] == 0].index
print("\nNumber of True class samples:", len(true_indices))
print("Number of False class samples:", len(false_indices))

# Split True class into 70-15-15
true_train_idx, true_temp_idx = train_test_split(
    true_indices, test_size=0.3, random_state=42, stratify=y[true_indices])
true_val_idx, true_test_idx = train_test_split(
    true_temp_idx, test_size=0.5, random_state=42, stratify=y[true_temp_idx])

# Match the False class size with True class for train and validation
false_train_idx = false_indices[:len(true_train_idx)]
false_val_idx = false_indices[len(true_train_idx):len(
    true_train_idx) + len(true_val_idx)]
false_test_idx = false_indices[len(true_train_idx) + len(true_val_idx):]

# Combine True and False indices for each split
train_idx = true_train_idx.union(false_train_idx)
val_idx = true_val_idx.union(false_val_idx)
test_idx = true_test_idx.union(false_test_idx)

# Split the data based on these indices
X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

# Assign 'sampled' labels
data['sampled'] = 'test'
data.loc[train_idx, 'sampled'] = 'train'
data.loc[val_idx, 'sampled'] = 'validation'

# Check if indices are valid
for idx_name, indices in zip(['train_index', 'val_index', 'test_index'], [train_idx, val_idx, test_idx]):
    if not set(indices).issubset(data.index):
        not_found = set(indices) - set(data.index)
        raise KeyError(f"{not_found} not found in the data for {idx_name}")

# Output class distribution for each split
print("\nClass distribution in train set:")
print(data[data["one_vs_rest"] == "train"]["binary_score"].value_counts())

print("\nClass distribution in validation set:")
print(data[data["one_vs_rest"] == "validation"]["binary_score"].value_counts())

print("\nClass distribution in test set:")
print(data[data["one_vs_rest"] == "test"]["binary_score"].value_counts())

data['aa_seq'] = data.apply(lambda x: wildtype[:mutated_region_start] + x['AASeq'] + wildtype[mutated_region_end:], axis=1)


# Calculate and add a new column for the length of each amino acid sequence
data["len"] = data["aa_seq"].apply(len)

# Creating a new DataFrame with the specified columns
new_data = pd.DataFrame(
    {
        "aa_seq": data["aa_seq"],
        "len": data["len"],
        "no_mut": data["no_mut"],
        "binary_score": data["binary_score"],
        "one_vs_rest": data["one_vs_rest"],
        "sampled": data["sampled"],
    }
)

new_data.drop_duplicates(subset="aa_seq", keep="first", inplace=True)

# Save the new DataFrame to a CSV file
new_data.to_csv(os.path.join(script_dir, "herH3_data_full.csv"), index=False)
