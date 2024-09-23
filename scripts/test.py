import blosum as bl
import numpy as np

# List of protein sequences
protein_sequences = ["AGTKG", "ILHFM", "CYQTE", "SDGGR", "HKLFW"]
BLOSUM62 = bl.BLOSUM(62) 

# List of protein sequences
protein_sequences = ["AGTKG", "ILHFM", "CYQTE", "SDGGR", "HKLFW"]

# Function to encode protein sequences using the BLOSUM62 matrix
def encode_sequences(sequences, matrix):
    encoded_sequences = []
    for seq in sequences:
        encoded_seq = []
        for acid in seq:
            row = matrix[acid]
            encoded_row = [row[aa] for aa in seq]  # Extract scores for the sequence from the row corresponding to each amino acid
            encoded_seq.append(encoded_row)
        encoded_sequences.append(np.array(encoded_seq))
    return encoded_sequences

# Encode the sequences
encoded_protein_sequences = encode_sequences(protein_sequences, BLOSUM62)

# Print the encoded sequences
for seq, encoded in zip(protein_sequences, encoded_protein_sequences):
    print(f"Original: {seq}\nEncoded:\n{encoded}\n")
