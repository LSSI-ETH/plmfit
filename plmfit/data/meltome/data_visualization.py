import pandas as pd
import shared_utils.data_explore as data_explore
import os

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

# Define file paths for the CSV and FASTA files
csv_path = "mixed_split.csv"

# Read the dataset from the CSV file
data = pd.read_csv(os.path.join(script_dir, csv_path))

if __name__ == "__main__":

    # Print the shape of the dataset
    print(data.shape)

    # Add a new column to the DataFrame for sequence length
    data["sequence_length"] = data["sequence"].apply(len)
    for i in range(len(data["sequence_length"])):
        if (data["sequence_length"][i] > 1000):
            print(data["sequence_length"][i])

    # Various visualization utility function calls
    data_explore.plot_score_distribution(
        data, column="target", path=os.path.join(script_dir, "plots/score.png"))
    data_explore.plot_sequence_length_distribution(
        data, path=os.path.join(script_dir, "plots/seq_len.png"))
