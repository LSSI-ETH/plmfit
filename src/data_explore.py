import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_label_distribution(data, label="binary_score"):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=label, data=data)
    plt.title('Label Distribution')
    plt.xlabel(label)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show(block=False)

def plot_score_distribution(data, column="score"):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], bins=50, kde=True)
    plt.title('Score Distribution')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show(block=False)

def normalized_score(data, column="score"):
    # Calculate the minimum and maximum values of the score column
    min_score = data[column].min()
    max_score = data[column].max()

    # Apply Min-Max Normalization
    return (data[column] - min_score) / (max_score - min_score)

def plot_normalized_score_distribution(data, column="normalized_score"):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], bins=50, kde=True)
    plt.title('Normalized Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.show(block=False)

def plot_sequence_length_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['sequence_length'], kde=True)
    plt.title('Length Distribution of Sequence')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.show(block=False)

def plot_mutations_number(data, column='number_of_mutations'):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True)
    plt.title('Number of Mutations Distribution')
    plt.xlabel('Number of Mutations')
    plt.ylabel('Frequency')
    plt.show(block=False)

def parse_fasta(fasta_file, log=False):
    with open(fasta_file, 'r') as file:
        sequence_id = ''
        sequence = ''
        for line in file:
            if line.startswith('>'):
                sequence_id = line[1:].strip()  # Removes the '>' and any trailing newline characters
            else:
                sequence += line.strip()  # Adds the sequence line, removing any trailing newlines
        if log:
            print("Sequence ID:", sequence_id)
            print("Sequence:", sequence)
        return sequence_id, sequence
    
def plot_mutations_heatmap(mutation_counts):
    # Convert the mutation counts to a DataFrame
    mutation_df = pd.DataFrame(mutation_counts)

    plt.figure(figsize=(15, 10))  # Adjust the figure size as needed
    sns.heatmap(np.transpose(mutation_df), cmap='viridis', cbar=True)
    plt.title('Mutation Heatmap per Amino Acid and Position')
    plt.xlabel('Position in Sequence')
    plt.ylabel('Amino Acids')
    plt.show()