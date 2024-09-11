import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

csv_path = 'rbd_data_full.csv'

# Imports the dataset
data = pd.read_csv(os.path.join(script_dir, csv_path))

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
false_test_idx = false_test_idx[:2*len(true_test_idx)]

# Combine True and False indices for each split
train_idx = true_train_idx.union(false_train_idx)
val_idx = true_val_idx.union(false_val_idx)
test_idx = true_test_idx.union(false_test_idx)

# Split the data based on these indices
X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

# Assign 'sampled' labels
data['sampled'] = ''
data.loc[train_idx, 'sampled'] = 'train'
data.loc[val_idx, 'sampled'] = 'validation'
data.loc[test_idx, 'sampled'] = 'test'

# Check if indices are valid
for idx_name, indices in zip(['train_index', 'val_index', 'test_index'], [train_idx, val_idx, test_idx]):
    if not set(indices).issubset(data.index):
        not_found = set(indices) - set(data.index)
        raise KeyError(f"{not_found} not found in the data for {idx_name}")

# Output class distribution for each split
print("\nClass distribution in train set:")
print(data[data['sampled'] == 'train']['binary_score'].value_counts())

print("\nClass distribution in validation set:")
print(data[data['sampled'] == 'validation']['binary_score'].value_counts())

print("\nClass distribution in test set:")
print(data[data['sampled'] == 'test']['binary_score'].value_counts())

# Export to csv
data.to_csv(os.path.join(script_dir, csv_path), index=False)
