import numpy as np
import pandas as pd
import os
import plmfit.shared_utils.utils as utils
from tqdm import tqdm
from plmfit.shared_utils.data_explore import plot_label_distribution
from plmfit.shared_utils.data_explore import plot_mutations_number
from plmfit.shared_utils.data_explore import plot_mutations_heatmap
import plmfit.shared_utils.data_explore as data_explore
import ast

# Directory of the script to be used while saving
script_dir = os.path.dirname(os.path.realpath(__file__))

# Read the paratope dataset from the csv file
data = pd.read_csv(os.path.join(script_dir, "paratope_data_full.csv"))
sequences = data["aa_seq"]
max_length = max(data['len'].values)
seq_count = len(sequences)

# Count the number of paratope positions = number of 1s in the label array
paratope_positions = data["label"]
paratope_positions = paratope_positions.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
paratope_positions = paratope_positions.apply(lambda x: sum(float(i) for i in x))
data["no_pos"] = paratope_positions

# Create the "plots" directory in case it doesn't exist
os.makedirs(os.path.join(script_dir, "plots"), exist_ok = True)

# Plot for the sequence length distribution
data_explore.plot_sequence_length_distribution(
    data, path=os.path.join(script_dir, "plots/seq_len.png")
)

# Plot for the distribution of number of paratope positions
data_explore.plot_label_distribution_token_level_classification(
    data, path=os.path.join(script_dir, "plots/token_labels.png")
)