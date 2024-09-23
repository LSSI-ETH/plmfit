import pandas as pd
import plmfit.shared_utils.data_explore as data_explore
import os

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

# Define file paths for the CSV and FASTA files
csv_path = "four_mutations_full_data.csv"
fasta_path = "5LDE_1.fasta"

# Read the dataset from the CSV file
data = pd.read_csv(os.path.join(script_dir, csv_path))


# Legacy function, might remove in the future
def get_pos_mut(seq, wildtype_seq):
    ret = []
    for i in range(len(seq)):
        if seq[i] != wildtype_seq[i]:
            ret.append(i)
            ret.append(seq[i])
    if len(ret) == 0:
        ret = [-1, '-']

    if len(ret) > 2:
        ret = [-2, '@']
    return ret

# Function to update mutation counts based on the wildtype sequence, mutation sequence, and mask


def update_mutation_counts(mutation_counts, wildtype_seq, mask, mutation_sites, log=False):
    insertions = 0
    deletions = 0
    region = ""
    for i in mutation_sites:
        seq_char = wildtype_seq[i]
        region += mask[i]
        if mask[i] != seq_char:  # Exclude no change and deletions
            mutation_counts[mask[i]][i] += 1

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


if __name__ == "__main__":

    # Define the indices for the mutation sites in the protein sequence according to the paper
    v39 = 38
    d40 = 39
    g41 = 40
    v54 = 53

    # List containing the mutation site indices
    mutation_sites = [v39, d40, g41, v54]

    # Print the shape of the dataset
    print(data.shape)

    # Add a new column to the DataFrame for sequence length
    data["sequence_length"] = data["sequence"].apply(len)

    # Find the maximum sequence length in the dataset for use in the mutation matrix
    max_length = max(data["sequence"].apply(len))

    # Initialize a dictionary to count mutations at each position for all amino acids
    mutation_counts = {aa: [0] * max_length for aa in "+-ACDEFGHIKLMNPQRSTVWY"}

    # Parse the FASTA file to get the sequence ID and sequence
    sequence_id, sequence = data_explore.parse_fasta(
        os.path.join(script_dir, fasta_path)
    )

    # Various visualization utility function calls
    data_explore.plot_label_distribution(
        data, label="keep", path=os.path.join(script_dir, "plots/labels.png"))
    data_explore.plot_score_distribution(
        data, column="Fitness", path=os.path.join(script_dir, "plots/score.png"))
    data["normalized_score"] = data_explore.normalized_score(
        data, column="Fitness"
    )  # Normalize score first
    data_explore.plot_normalized_score_distribution(
        data, log_scale=True, path=os.path.join(script_dir, "plots/norm_score.png"))
    data_explore.plot_sequence_length_distribution(
        data, path=os.path.join(script_dir, "plots/seq_len.png"))
    data_explore.plot_mutations_number(
        data, column="HD", annotation=True, path=os.path.join(script_dir, "plots/mut_no.png"))

    # Apply the function to update mutation counts for each row in the DataFrame
    data.apply(
        lambda row: update_mutation_counts(mutation_counts, sequence, row["sequence"], mutation_sites), axis=1
    )

    # Plot a heatmap of mutations
    data_explore.plot_mutations_heatmap(mutation_counts, zoom_region=[
                                        35, 60], path=os.path.join(script_dir, "plots/mut_heatmap.png"))
