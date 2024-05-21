import pandas as pd
import os
import json
import numpy as np

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

# Define file paths for the CSV and FASTA files
csv_path = "mixed_split.csv"

# Read the dataset from the CSV file
data = pd.read_csv(os.path.join(script_dir, csv_path))

if __name__ == "__main__":

    # Add a new column to the DataFrame for sequence length
    data["sequence_length"] = data["sequence"].apply(len)

    data["mixed"] = data["set"]
    data["mixed"] = np.where(data["validation"].isna(), data['set'], 'validation')

    # Create a new DataFrame with specified columns and save it as a CSV file
    new_data = pd.DataFrame(
        {
            "aa_seq": data["sequence"],
            "len": data["sequence_length"],
            "score": data["target"],
            "mixed": data["mixed"],
        }
    )

    
    print(len(new_data))

    new_data = new_data[~new_data["aa_seq"].str.contains("\*")]
    new_data.sort_values("score", inplace=True)
    new_data.drop_duplicates(subset="aa_seq", keep="first", inplace=True)
    print(len(new_data))
    # Drop all entries with a sequence length greater than 1000
    # Count the number of sequences below different sequence lengths
    for length in range(250, 4001, 250):
        count = len(new_data[(new_data["len"] <= length) & (new_data["mixed"] == "train")])
        print(f"Number of sequences below {length}: {count}")

    # Filter the new_data DataFrame to keep sequences with length <= 800
    new_data = new_data[new_data["len"] <= 750]

    # Save the new DataFrame to a CSV file
    new_data.to_csv(os.path.join(
        script_dir, "meltome_data_full.csv"), index=False)
