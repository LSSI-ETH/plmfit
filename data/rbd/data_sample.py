import pandas as pd
import plmfit.shared_utils.data_explore as data_explore
import os
import json
import numpy as np

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

# File paths for the dataset and FASTA file
csv_path = "rbd_data_full.csv"

# Load dataset from CSV file
data = pd.read_csv(
    os.path.join(script_dir, csv_path)
) 

# Sample 10k sequences out of the dataset trying to keep the same distribution
data = data.sample(n=10000, random_state=1)

data.to_csv(os.path.join(script_dir, "rbd_data_sample.csv"), index=False)
