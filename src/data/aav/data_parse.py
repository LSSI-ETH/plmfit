import pandas as pd
from src import data_explore
import os
import json

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

    # File paths for the dataset and FASTA file
    csvPath = "full_data.csv"
    fastaPath = "P03135.fasta"

    # Load dataset from CSV file
    data = pd.read_csv(
        os.path.join(script_dir, csvPath), dtype={"one_vs_many_split_validation": float}
    )  # solves DtypeWarning: Columns have mixed types. Specify dtype option on import or set low_memory=False in Pandas

    # Print the shape of the dataset
    print(data.shape)

    # Calculate and add a new column for the length of each amino acid sequence
    data["sequence_length"] = data["full_aa_sequence"].apply(len)

    # Determine the maximum length among all sequences
    max_length = max(data["full_aa_sequence"].apply(len))

    # Initialize a dictionary to count mutations for each amino acid
    mutation_counts = {aa: [0] * max_length for aa in "+-ACDEFGHIKLMNPQRSTVWY"}

    # Parse the FASTA file to extract sequence ID and sequence
    sequence_id, sequence = data_explore.parse_fasta(
        os.path.join(script_dir, fastaPath)
    )

    # Various plots for data visualization
    data_explore.plot_label_distribution(data)
    data_explore.plot_score_distribution(data)

    # Normalize the scores and plot the distribution
    data["normalized_score"] = data_explore.normalized_score(data)
    data_explore.plot_normalized_score_distribution(data)

    # Plot distributions of sequence lengths and number of mutations
    data_explore.plot_sequence_length_distribution(data)
    data_explore.plot_mutations_number(data)

    # Function to update mutation counts based on the wild type sequence, mutation region, and mask
    def update_mutation_counts(wildtype_seq, region, mask, log=False):
        start_pos = wildtype_seq.find(
            region
        )  # Find the starting position of the mutated region
        end_pos = start_pos + len(region)  # Ending position of the mutated region

        # Initialize counts for insertions and deletions
        insertions = 0
        deletions = 0

        # Update mutation counts, aligning with the full sequence
        j = 0
        for i in range(start_pos, end_pos):
            seq_char = wildtype_seq[i]  # Current amino acid in the wildtype sequence

            # Corresponding character in the mutation mask, if mask is smaller than the region just use padding
            if len(mask) > j:
                mask_char = mask[j]
            else:
                mask_char = "_"

            # If we encounter insertion or deletion, document it and move to the next aminoacid to avoid ruining the positions of the mutation matrix
            while mask_char.islower() or mask_char == "*":
                if mask_char.islower():  # Insertion
                    insertions += 1
                    mutation_counts["+"][i] += 1
                if mask_char == "*":  # Deletion
                    deletions += 1
                    mutation_counts["-"][i] += 1
                j += 1
                if len(mask) > j:
                    mask_char = mask[j]
                else:
                    mask_char = "_"
            if mask_char != "_":  # Exclude no change and deletions
                mutation_counts[seq_char][i] += 1  # Change
            j += 1

        # Optional logging
        if log:
            print(
                mask,
                " - Length: ",
                len(mask),
                " - Deletions: ",
                deletions,
                " - Insertions: ",
                insertions,
            )
            print(region, " - Length: ", len(region))
            print("\n")

    # Apply the mutation count update function to each row in the dataset
    data.apply(
        lambda row: update_mutation_counts(
            sequence, row["reference_region"], row["mutation_mask"]
        ),
        axis=1,
    )

    # Plot a heatmap of mutations
    data_explore.plot_mutations_heatmap(mutation_counts)

    # Creating a new DataFrame with the specified columns
    new_data = pd.DataFrame(
        {
            "aa_seq": data["full_aa_sequence"],
            "len": data["sequence_length"],
            "no_mut": data["number_of_mutations"],
            "score": data["normalized_score"],
        }
    )
    new_data = new_data[~new_data['aa_seq'].str.contains('\*')]
    new_data.drop_duplicates(subset="aa_seq", keep="first", inplace=True)

    # Save the new DataFrame to a CSV file
    new_data.to_csv(os.path.join(script_dir, "aav_data_full.csv"), index=False)

    # Define the JSON file path
    json_file_path = 'wild_type.json'

    wildtype = {
        "wild_type": sequence,
        "meta": sequence_id
    }

    # Write the data to the JSON file
    with open(os.path.join(script_dir, json_file_path), 'w') as json_file:
        json.dump(wildtype, json_file, indent=4)
