import pandas as pd
from src import data_explore
import os
import json

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

# Define file paths for the CSV and FASTA files
csv_path = "mixed_split.csv"

# Read the dataset from the CSV file
data = pd.read_csv(os.path.join(script_dir, csv_path))

if __name__ == "__main__":

    # Add a new column to the DataFrame for sequence length
    data["sequence_length"] = data["sequence"].apply(len)


    # Create a new DataFrame with specified columns and save it as a CSV file
    new_data = pd.DataFrame(
        {
            "aa_seq": data["sequence"],
            "len": data["sequence_length"],
            "score": data["target"],
            "set": data["set"],
            "set_with_val": data.apply(lambda row: "val" if row["validation"] == True else row["set"], axis=1)
        }
    )

    new_data = new_data[~new_data["aa_seq"].str.contains("\*")]
    new_data.drop_duplicates(subset="aa_seq", keep="first", inplace=True)

    # Drop all entries with a sequence length greater than 1000
    new_data = new_data[new_data["len"] <= 1000]

    # Save the new DataFrame to a CSV file
    new_data.to_csv(os.path.join(script_dir, "meltome_data_full.csv"), index=False)
