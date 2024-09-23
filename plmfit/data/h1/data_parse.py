import os
from Bio import SeqIO  # BioPython for reading FASTA files
import pandas as pd

def read_fasta(fasta_path):
    """
    Read a FASTA file and return a list of sequences.
    """
    sequences = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append({
            "aa_seq": str(record.seq)  # Store the amino acid sequence as a string
        })
    return sequences

def filter_sequences(sequences, max_length=1000, min_length=0):
    """
    Filter sequences to remove those containing '*' and those longer than max_length.
    """
    filtered_data = pd.DataFrame(sequences)
    filtered_data["len"] = filtered_data["aa_seq"].apply(len)  # Calculate sequence length
    filtered_data = filtered_data[~filtered_data["aa_seq"].str.contains("\*")]  # Remove sequences containing '*'
    filtered_data = filtered_data[~filtered_data["aa_seq"].str.contains("X")]  # Remove sequences containing 'X'
    filtered_data = filtered_data[filtered_data["len"] <= max_length]  # Limit sequences to max_length
    filtered_data = filtered_data[filtered_data["len"] >= min_length]  # Limit sequences to min_length
    return filtered_data

def remove_duplicates(data):
    """
    Remove duplicate sequences from the DataFrame.
    """
    unique_data = data.drop_duplicates(subset="aa_seq", keep='first', inplace=False)  # Drop duplicate sequences
    return unique_data

def main():
    script_dir = os.path.dirname(__file__)  # Absolute dir the script is in
    fasta_path = "H1_sequences.fasta"  # Path to your FASTA file

    # Read the dataset from the FASTA file
    sequences = read_fasta(os.path.join(script_dir, fasta_path))

    # Filter sequences and prepare DataFrame
    filtered_data = filter_sequences(sequences, max_length=566, min_length=566)

    # Remove duplicate sequences
    unique_data = remove_duplicates(filtered_data)

    # Save the filtered and unique data to a CSV file
    csv_output_path = os.path.join(script_dir, "h1_data_full.csv")
    unique_data.to_csv(csv_output_path, index=False)
    print(f"Processed data saved to {csv_output_path}")

if __name__ == "__main__":
    main()
