import pandas as pd
import os
import json
import numpy as np
from Bio import SeqIO
from plmfit.shared_utils import data_explore

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

# File paths for the FASTA files
sequences_path = "sequences.fasta"
sampled_path = "sampled.fasta"
mask_path = "mask.fasta"

# Load the sequences to extract the sequence ID and sequence
with open(os.path.join(script_dir, sequences_path), "r") as file:
    sequences = []
    for record in SeqIO.parse(file, "fasta"):
        sequences.append([record.name, str(record.seq)])

    # Create dataframe
    data = pd.DataFrame(sequences, columns=["name", "aa_seq"])

with open(os.path.join(script_dir, sampled_path), "r") as file:
    sequences = []
    for record in SeqIO.parse(file, "fasta"):

        sequences.append([str(record.seq), record.description])

    # Add to dataframe
    data = pd.concat([data, pd.DataFrame(sequences, columns=["label", "sampled"])], axis=1)

with open(os.path.join(script_dir, mask_path), "r") as file:
    sequences = []
    for record in SeqIO.parse(file, "fasta"):
        sequences.append([str(record.seq)])

    # Add to dataframe
    data = pd.concat([data, pd.DataFrame(sequences, columns=["mask"])], axis=1)

# Get data split information
data["validation"] = data.sampled.str.split("=").str[2]
# str to bool
data["validation"] = data["validation"].apply(lambda x: x == "True")

# Extract data split information
data["sampled"] = data.sampled.str.split("=").str[1]
data["sampled"] = data.sampled.str.split(" ").str[0]

# If validation is True, then it is in the validation set and add on sampled column the word 'validation'
data["sampled"] = np.where(data["validation"], "validation", data["sampled"])

data.drop(columns=["validation"], inplace=True)

data["len"] = data["aa_seq"].apply(lambda x: len(x))

# Preprocess mask and label to lists
# C is class 0, E is class 1, H is class 2
data["label"] = data["label"].str.replace("C", "0")
data["label"] = data["label"].str.replace("E", "1")
data["label"] = data["label"].str.replace("H", "2")

# str to integer
data["label"] = data["label"].apply(lambda x: [int(i) for i in x])
data["mask"] = data["mask"].apply(lambda x: [int(i) for i in x])

# Print length of sequences
print("Length of sequences:")
print(data["len"].describe())

# Print set distribution
print("\nSet distribution:")
print(data["sampled"].value_counts())

# Calculate class distribution. For each list of labels (0, 1, 2), count the number of times each label appears only if the mask is 1 and then print the total count for each label
class_distribution = data.apply(lambda x: [x["label"][i] for i in range(len(x["label"])) if x["mask"][i] == 1], axis=1)
class_distribution = class_distribution.explode().value_counts()
print("\nClass distribution:")
print(class_distribution)

# Plot distributions of sequence lengths and number of mutations
data_explore.plot_sequence_length_distribution(
    data, path=os.path.join(script_dir, "plots/seq_len.png")
)


# Save the new DataFrame to a CSV file
data.to_csv(os.path.join(script_dir, "ss3_data_full.csv"), index=False)
