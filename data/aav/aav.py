import pandas as pd
import visualization_utils

# File paths for the dataset and FASTA file
csvPath = 'full_data.csv'
fastaPath = 'P03135.fasta'

# Load dataset from CSV file
data = pd.read_csv(csvPath, dtype={'one_vs_many_split_validation': float})

# Print the shape of the dataset
print(data.shape)

# Calculate and add a new column for the length of each amino acid sequence
data['sequence_length'] = data['full_aa_sequence'].apply(len)

# Determine the maximum length among all sequences
max_length = max(data['full_aa_sequence'].apply(len))

# Initialize a dictionary to count mutations for each amino acid
mutation_counts = {aa: [0] * max_length for aa in '+-ACDEFGHIKLMNPQRSTVWY'}

# Parse the FASTA file to extract sequence ID and sequence
sequence_id, sequence = visualization_utils.parse_fasta(fastaPath)

# Various plots for data visualization
visualization_utils.plot_label_distribution(data)
visualization_utils.plot_score_distribution(data)

# Normalize the scores and plot the distribution
data['normalized_score'] = visualization_utils.normalized_score(data)
visualization_utils.plot_normalized_score_distribution(data)

# Plot distributions of sequence lengths and number of mutations
visualization_utils.plot_sequence_length_distribution(data)
visualization_utils.plot_mutations_number(data)

# Function to update mutation counts based on the wild type sequence, mutation region, and mask
def update_mutation_counts(wildtype_seq, region, mask, log=False):
    start_pos = wildtype_seq.find(region)  # Find the starting position of the mutated region
    end_pos = start_pos + len(region)  # Ending position of the mutated region
    
    # Initialize counts for insertions and deletions
    insertions = 0
    deletions = 0
    
    # Update mutation counts, aligning with the full sequence
    j = 0
    for i in range(start_pos, end_pos):
        seq_char = wildtype_seq[i] # Current amino acid in the wildtype sequence
        
        # Corresponding character in the mutation mask, if mask is smaller than the region just use padding
        if len(mask) > j:
            mask_char = mask[j]
        else:
            mask_char = '_'
            
        # If we encounter insertion or deletion, document it and move to the next aminoacid to avoid ruining the positions of the mutation matrix
        while mask_char.islower() or mask_char == '*':
            if mask_char.islower(): # Insertion
                insertions += 1
                mutation_counts['+'][i] += 1
            if mask_char == '*': # Deletion
                deletions += 1
                mutation_counts['-'][i] += 1
            j += 1
            if len(mask) > j:
                mask_char = mask[j]
            else:
                mask_char = '_'
        if mask_char != '_':  # Exclude no change and deletions
            mutation_counts[seq_char][i] += 1 # Change
        j += 1
        
    # Optional logging
    if log: 
        print(mask, " - Length: ", len(mask), " - Deletions: ", deletions, " - Insertions: ", insertions)
        print(region, " - Length: ", len(region))
        print("\n")

# Apply the mutation count update function to each row in the dataset
data.apply(lambda row: update_mutation_counts(sequence, row['reference_region'], row['mutation_mask']), axis=1)

# Plot a heatmap of mutations
visualization_utils.plot_mutations_heatmap(mutation_counts)

# Creating a new DataFrame with the specified columns
new_data = pd.DataFrame({
    'aa_seq': data['full_aa_sequence'],
    'len': data['sequence_length'],
    'no_mut': data['number_of_mutations'],
    'score': data['normalized_score']
})


# Save the new DataFrame to a CSV file
new_data.to_csv('processed_data.csv', index=False)


