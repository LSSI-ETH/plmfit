# Import required libraries
import pandas as pd
import visualization_utils

# Define file paths for the CSV and FASTA files
csvPath = 'four_mutations_full_data.csv'
fastaPath = '5LDE_1.fasta'

# Read the dataset from the CSV file
data = pd.read_csv(csvPath)

# Define the indices for the mutation sites in the protein sequence according to the paper
v39 = 38
d40 = 39
g41 = 40
v54 = 53

# List containing the mutation site indices
mutationSites = [v39, d40, g41, v54]

# Print the shape of the dataset
print(data.shape)

# Add a new column to the DataFrame for sequence length
data['sequence_length'] = data['sequence'].apply(len)

# Find the maximum sequence length in the dataset for use in the mutation matrix
max_length = max(data['sequence'].apply(len))

# Initialize a dictionary to count mutations at each position for all amino acids
mutation_counts = {aa: [0] * max_length for aa in '+-ACDEFGHIKLMNPQRSTVWY'}

# Parse the FASTA file to get the sequence ID and sequence
sequence_id, sequence = visualization_utils.parse_fasta(fastaPath)

# Various visualization utility function calls
visualization_utils.plot_label_distribution(data, label="keep")
visualization_utils.plot_score_distribution(data, column="Fitness")
data['normalized_score'] = visualization_utils.normalized_score(data, column="Fitness") # Normalize score first
visualization_utils.plot_normalized_score_distribution(data)
visualization_utils.plot_sequence_length_distribution(data)
visualization_utils.plot_mutations_number(data, column="HD")

# Function to update mutation counts based on the wildtype sequence, mutation sequence, and mask
def update_mutation_counts(wildtype_seq, mutation_seq, mask, log=False):
    insertions = 0
    deletions = 0
    region = ''
    j = 0
    for i in mutationSites:
        seq_char = wildtype_seq[i]
        region += seq_char
        if mask[j] != seq_char:  # Exclude no change and deletions
            mutation_counts[mask[j]][i] += 1
        j += 1
    
    # Optional logging
    if log: 
        print(mask, " - Length: ", len(mask), " - Deletions: ", deletions, " - Insertions: ", insertions)
        print(region, " - Length: ", len(region))
        print("\n")

# Apply the function to update mutation counts for each row in the DataFrame
data.apply(lambda row: update_mutation_counts(sequence, row['sequence'], row['Variants'],  log=True), axis=1)

# Plot a heatmap of mutations
visualization_utils.plot_mutations_heatmap(mutation_counts)

# Create a new DataFrame with specified columns and save it as a CSV file
new_data = pd.DataFrame({
    'aa_seq': data['sequence'],
    'len': data['sequence_length'],
    'no_mut': data['HD'],
    'score': data['normalized_score']
})

# Save the new DataFrame to a CSV file
new_data.to_csv('processed_data.csv', index=False)
