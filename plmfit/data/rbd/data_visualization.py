import numpy as np
import pandas as pd
import os
import plmfit.shared_utils.utils as utils
from tqdm import tqdm
from plmfit.shared_utils.data_explore import plot_label_distribution
from plmfit.shared_utils.data_explore import plot_mutations_number
from plmfit.shared_utils.data_explore import plot_mutations_heatmap

# Directory of the script to be used while saving
script_dir = os.path.dirname(os.path.realpath(__file__))

# Calculation of the mutation counts and building of the mutation dictionary
data = utils.load_dataset("rbd")
sequences = data["aa_seq"]
max_length = max(data['len'].values) # All sequences have length 201
seq_count = len(sequences)
mut_counts = np.zeros(seq_count) # Number of mutations in a given sequence
wild_type = utils.get_wild_type("rbd")
sequences = utils.load_dataset("rbd")["aa_seq"]

# A dictionary in which each amino acid is a "key", and has a "value"
# of a numpy array of sequence length, where an index is number of mutations
mut_dict = {aa: np.array([0] * max_length) for aa in "ACDEFGHIKLMNPQRSTVWY"}
# Sequence is converted to a numpy array for easy comparison
wild_list = str_list = np.array([aa for aa in wild_type])

for i in tqdm(range(seq_count)):
    seq = sequences[i]
    
    # Sequence is converted to a numpy array for easy comparison
    seq_list = np.array([aa for aa in seq])
    comparison = wild_list!=seq_list # A matrix where mutations are "True"
    mut_counts[i] = np.sum(comparison) # Sum of true values = mutation count
    
    # For each mutated position, mutation count of the AA is incremented
    mut_positions = np.where(comparison) 
    for pos in mut_positions:
        aa = seq_list[pos][0]
        mut_dict[aa][pos] += 1

# Create the "plots" directory in case it doesn't exist
os.makedirs(os.path.join(script_dir, "plots"), exist_ok = True)

# Plot for label distribution
data["label"] = data["label"].map(bool)
save_path =  os.path.join(script_dir, "plots\\labels.png")
plot_label_distribution(data, label="label", path=save_path, text="Binding")

# Plot for distribution of number of mutations
mut_count_data = pd.DataFrame({"number_of_mutations":mut_counts})
save_path =  os.path.join(script_dir, "plots\\mut_no.png")
plot_mutations_number(mut_count_data, annotation=False, path=save_path)

# Plot for mutation heat map
save_path =  os.path.join(script_dir, "plots\\mut_heat_map")
plot_mutations_heatmap(mut_dict,zoom_region=[60,90], path=save_path)