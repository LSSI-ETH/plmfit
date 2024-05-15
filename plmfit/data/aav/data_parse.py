import pandas as pd
import plmfit.shared_utils.data_explore as data_explore
import os
import json
import numpy as np

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

# File paths for the dataset and FASTA file
csv_path = "full_data.csv"
fasta_path = "P03135.fasta"

# Load dataset from CSV file
data = pd.read_csv(
    os.path.join(script_dir, csv_path), dtype={"one_vs_many_split_validation": float}
)  # solves DtypeWarning: Columns have mixed types. Specify dtype option on import or set low_memory=False in Pandas


# Function to update mutation counts based on the wild type sequence, mutation region, and mask
def get_mutation_positions(wildtype_seq, region, mask):
    start_pos = wildtype_seq.find(
        region
    )  # Find the starting position of the mutated region
    end_pos = len(mask)  # Ending position of the mutated region
    positions = []
    deletions = 0 # Track deletions because we need to subtract one positions for each deletion
    for i in range(end_pos):
        mask_char = mask[i]
        if mask_char != '_':
            if mask_char == '*':
                deletions = deletions + 1
            else:
                positions.append(i + start_pos - deletions)
    return positions

if __name__ == "__main__":
    data = data[~data['sampled_split'].isna()]
    data["two_vs_many_split"] = np.where(data["two_vs_many_split_validation"].isna(), data['two_vs_many_split'], 'validation')

    # Calculate and add a new column for the length of each amino acid sequence
    data["sequence_length"] = data["full_aa_sequence"].apply(len)

    # Parse the FASTA file to extract sequence ID and sequence
    sequence_id, sequence = data_explore.parse_fasta(
        os.path.join(script_dir, fasta_path)
    )

    # Normalize the scores and plot the distribution
    data["normalized_score"] = data_explore.normalized_score(data)


    # Creating a new DataFrame with the specified columns
    new_data = pd.DataFrame(
        {
            "aa_seq": data["full_aa_sequence"],
            "len": data["sequence_length"],
            "no_mut": data["number_of_mutations"],
            "score": data["normalized_score"],
            "binary_score": data["binary_score"],
            "two_vs_many": data["two_vs_many_split"],
            "mut_mask":  data.apply(lambda row: get_mutation_positions(sequence, row["reference_region"], row["mutation_mask"]),axis=1)
        }
    )
    
    new_data = new_data[~new_data["aa_seq"].str.contains("\*")]
    new_data.drop_duplicates(subset="aa_seq", keep="first", inplace=True)

    # Save the new DataFrame to a CSV file
    new_data.to_csv(os.path.join(script_dir, "aav_data_full.csv"), index=False)

    # Define the JSON file path
    json_file_path = "wild_type.json"

    wildtype = {"wild_type": sequence, "meta": sequence_id}

    # Write the data to the JSON file
    with open(os.path.join(script_dir, json_file_path), "w") as json_file:
        json.dump(wildtype, json_file, indent=4)
