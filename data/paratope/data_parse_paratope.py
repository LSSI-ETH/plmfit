import pandas as pd
import os

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

# Load the training, testing, and validation data
data_train = pd.read_parquet(
    'sabdab_train.parquet'
)
data_test = pd.read_parquet(
    'sabdab_test.parquet'
)
data_val = pd.read_parquet(
    'sabdab_val.parquet'
)

if __name__ == "__main__":
    # Add the train, test, val labels to the data
    data_train["sampled"] = "train"
    data_test["sampled"] = "test"
    data_val["sampled"] = "validation"

    # Combine the training, validation, and test data
    data = pd.concat([data_train, data_test, data_val])                                                                            

    # Calculate and add a new column for the length of each amino acid sequence
    data["sequence_length"] = data["paratope_sequence"].apply(len)

    # Transform each paratope label into a binary array
    data["paratope_labels_binary"] = data["paratope_labels"].apply(
        lambda x: [0. if char == 'N' else 1. for char in x]
    )

    # Creating a new DataFrame with the specified columns
    new_data = pd.DataFrame(
        {
            "aa_seq": data["sequence"],
            "len": data["sequence_length"],
            "label": data["paratope_labels_binary"],
            "sampled": data["sampled"]
        }
    )

    # Save the new data frame as a CSV file
    new_data.to_csv(os.path.join(script_dir, "paratope_data_full.csv"), index=False)