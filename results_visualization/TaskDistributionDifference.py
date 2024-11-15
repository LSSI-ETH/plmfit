import numpy as np
from collections import Counter
from scipy.special import rel_entr

# Function to calculate the PSFM for a given set of sequences
def calculate_psfm(sequences):
    sequence_length = len(sequences[0])
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    
    # Initialize a PSFM matrix with zeros (positions x amino acids)
    psfm = np.zeros((sequence_length, len(amino_acids)))
    
    # For each position, calculate amino acid frequencies
    for pos in range(sequence_length):
        # Collect all amino acids at the current position
        amino_at_pos = [seq[pos] for seq in sequences]
        
        # Count the occurrences of each amino acid at this position
        counts = Counter(amino_at_pos)
        
        # Update the PSFM for this position
        for aa, count in counts.items():
            psfm[pos][amino_acids.index(aa)] = count / len(sequences)
    
    return psfm, amino_acids

# Function to compute KL divergence between two PSFMs
def calculate_kl_divergence(psfm1, psfm2):
    kl_divergences = []
    
    # Iterate over positions
    for pos in range(psfm1.shape[0]):
        # KL divergence is only computed if both distributions are non-zero
        p = psfm1[pos]
        q = psfm2[pos]
        
        # Add a small epsilon to avoid log(0) or divide-by-zero issues
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        
        # Calculate the KL divergence for this position
        kl_div = np.sum(rel_entr(p, q))  # rel_entr computes element-wise p * log(p / q)
        kl_divergences.append(kl_div)
    
    return kl_divergences

# Function to compute the average KL divergence between two sets of protein sequences
def compare_kl_divergence_avg(set1, set2):
    # Ensure sequences are of the same length or aligned
    if len(set1[0]) != len(set2[0]):
        raise ValueError("Sequences in both sets must be of the same length or aligned")
    
    # Calculate PSFM for both sets
    psfm1, amino_acids = calculate_psfm(set1)
    psfm2, _ = calculate_psfm(set2)
    
    # Compute the KL divergence for each position
    kl_divergences = calculate_kl_divergence(psfm1, psfm2)
    
    # Calculate the average KL divergence across all positions
    avg_kl_divergence = np.mean(kl_divergences)
    
    return avg_kl_divergence

# Example sequences for Set 1 and Set 2
set1 = ['ACDNWGTIKL', 'ACDEFGMIKL', 'ANDEFGIIKL', 'AEIGFGHKKL', 'ANPEFNHIKL']
set2 =  ['ACDEFYHIKL', 'ACDEFGHIQL', 'ACDDFGHIKL', 'ACVEFGHIKL', 'ACDMFGHIKL']
set3 = ['IHGILSIPYG', 'DGAHIGCIEH', 'GIVFPIFVKP']

# Compare the two sets using KL divergence
average_kl_divergence = compare_kl_divergence_avg(set2, set3)

# Print the average KL divergence
print(f"Average KL Divergence: {average_kl_divergence:.4f}")


import pandas as pd

# Creating a dummy pandas DataFrame to represent the approximated data points
data = {
    'Concentration (µM)': [0.002, 0.03, 0.1, 1, 3, 10, 30, 100],  # X values (approximate concentrations)
    'Response Units': [1, 3, 5, 9, 13, 20, 25, 30]  # Y values (approximate responses)
}

df = pd.DataFrame(data)
df
