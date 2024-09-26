import pandas as pd
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

# File paths for the dataset and FASTA file
tsv_path = "uniprotkb_AND_reviewed_true_2024_09_25.tsv"

# Load dataset from CSV file
data = pd.read_csv(os.path.join(script_dir, tsv_path), delimiter="\t")
print(data.head())

# Filter out duplicates based on sequence
data.drop_duplicates(subset="Sequence", keep="first", inplace=True)

print(f'All data: {data.shape[0]}')

# Filter out rows with non 'reviewed' values in the "Reviewed" column
data = data[data["Reviewed"] == "reviewed"]
print(f'Reviewed data: {data.shape[0]}')

# Filter out rows with missing values in the "EC number" column
data = data.dropna(subset=["EC number"])
print(f'EC number data: {data.shape[0]}')

# Find and remove all the rows with EC numbers that contain at least one ';' character
data["label"] = data["EC number"].apply(lambda x: x if ";" not in x else None)
data = data.dropna(subset=["label"])
print(f"After removing multiclass enzymes: {data.shape[0]}")

data["label"] = data["label"].str.split(".").str[0]
print(data["label"].value_counts())

print(data.head())

data["sampled"] = "train"

# For each class, add 15% to test and 15% to validation
for label in data["label"].unique():
    class_data = data[data["label"] == label]
    test_size = int(0.15 * len(class_data))
    validation_size = int(0.15 * len(class_data))

    test_indices = class_data.sample(test_size).index
    validation_indices = class_data.drop(test_indices).sample(validation_size).index

    data.loc[test_indices, "sampled"] = "test"
    data.loc[validation_indices, "sampled"] = "validation"

# Creating a new DataFrame with the specified columns
new_data = pd.DataFrame(
    {
        "aa_seq": data["Sequence"],
        "len": data["Length"],
        "label": data["label"],
        "sampled": data["sampled"],
    }
)

# Drop all entries with a sequence containing 'X' 'O' 'J' 'U' 'Z' 'B'
new_data = new_data[~new_data["aa_seq"].str.contains("X|O|J|U|Z|B")]
new_data.drop_duplicates(subset="aa_seq", keep="first", inplace=True)

for length in range(0, 1001, 100):
    count = len(new_data[(new_data["len"] <= length) & (new_data["sampled"] == "train")])
    print(f"Number of sequences below {length}: {count}")

# Filter the new_data DataFrame to keep sequences with length <= 500
new_data = new_data[new_data["len"] <= 500]

print(f"Final data: {new_data.shape[0]}")
print(new_data["label"].value_counts())
print(new_data["sampled"].value_counts())

labels = data["label"].values  # This extracts the labels from your data

# Compute class weights using 'balanced' mode, which assigns weights inversely proportional to the class frequencies
class_weights = compute_class_weight(
    "balanced", classes=np.unique(new_data["label"]), y=labels
)

# Convert the result into a dictionary mapping class labels to their corresponding weights
class_weights_dict = dict(zip(np.unique(labels), class_weights))

print(class_weights_dict)

# Add the class weights to the new_data DataFrame
new_data["weights"] = new_data["label"].map(class_weights_dict)

# Save the new DataFrame to a CSV file
new_data.to_csv(os.path.join(script_dir, "ezy1_data_full.csv"), index=False)
