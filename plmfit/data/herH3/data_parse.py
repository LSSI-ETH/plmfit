import pandas as pd
import plmfit.shared_utils.utils as utils
import os
import json
import numpy as np

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
dms = pd.DataFrame(data, columns=['AASeq', 'binary_score'])




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

data = pd.concat([pos_data, neg_data])

data.rename(columns={"AgClass": "binary_score"}, inplace=True)

data = pd.concat([dms, data])



# calculate no_mut column based on mutated region for each chain, no mask is used
data['no_mut'] = data['AASeq'].apply(lambda x: compare_strings(mutated_region, x))

data['one_vs_rest'] = np.where(data['no_mut'] <= 1, 'train', 'test')

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
        "one_vs_rest": data["one_vs_rest"]
    }
)

new_data.drop_duplicates(subset="aa_seq", keep="first", inplace=True)

# Save the new DataFrame to a CSV file
new_data.to_csv(os.path.join(script_dir, "herH3_data_full.csv"), index=False)


