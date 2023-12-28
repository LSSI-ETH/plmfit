import pandas as pd
from src import data_explore
import os
import json

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

# Define file paths for the CSV and FASTA files
csv_path = "four_mutations_full_data.csv"
fasta_path = "5LDE_1.fasta"

# Read the dataset from the CSV file
data = pd.read_csv(os.path.join(script_dir, csv_path))

if __name__ == "__main__":

    # Define the indices for the mutation sites in the protein sequence according to the paper
    v39 = 38
    d40 = 39
    g41 = 40
    v54 = 53

    # List containing the mutation site indices
    mutation_sites = [v39, d40, g41, v54]

    # Add a new column to the DataFrame for sequence length
    data["sequence_length"] = data["sequence"].apply(len)

    # Parse the FASTA file to get the sequence ID and sequence
    sequence_id, sequence = data_explore.parse_fasta(
        os.path.join(script_dir, fasta_path)
    )

    data["normalized_score"] = data_explore.normalized_score(
        data, column="Fitness"
    )  # Normalize score first

    # Create a new DataFrame with specified columns and save it as a CSV file
    new_data = pd.DataFrame(
        {
            "aa_seq": data["sequence"],
            "len": data["sequence_length"],
            "no_mut": data["HD"],
            "score": data["normalized_score"],
        }
    )

    new_data = new_data[~new_data["aa_seq"].str.contains("\*")]
    new_data.drop_duplicates(subset="aa_seq", keep="first", inplace=True)

    # Save the new DataFrame to a CSV file
    new_data.to_csv(os.path.join(script_dir, "gb1_data_full.csv"), index=False)

    # Define the JSON file path
    json_file_path = "wild_type.json"

    wildtype = {"wild_type": sequence, "meta": sequence_id}

    # Write the data to the JSON file
    with open(os.path.join(script_dir, json_file_path), "w") as json_file:
        json.dump(wildtype, json_file, indent=4)
