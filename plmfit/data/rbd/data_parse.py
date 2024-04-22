import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from plmfit.shared_utils import utils

"""
data_parse.py prepares rbd_data_full.csv for use with plmfit
- enters "./raw_data/raw_data" and searches for the relevant files (correct antibodies & cleaner data)
- cleans data, removes redundant and contradictory sequences
- creates rbd_data_full.csv
"""

if __name__ == '__main__':

    # Load ab dictionary file, initialize df and define folder_path
    antibody_dict = pd.read_csv("raw_data/antibody_dictionary.csv")

    data = pd.DataFrame(columns=['aa_seq', 'len', 'label', 'antibody'])
    folder_path = "raw_data/raw_data"

    # Collect data from files
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv") and ("Lib1_labeled" in file_name or "Lib2_labeled" in file_name):

            file_path = os.path.join(folder_path, file_name)
            temp_df = pd.DataFrame(columns=['aa_seq', 'len', 'label', 'antibody'])

            # Load CSV
            df = pd.read_csv(file_path)

            # Get rid of noise (low Total_sum) and irrelevant antibodies
            df = df[(df['Total_sum'] >= 2) & (df['Target'].isin(antibody_dict['Name']))]

            temp_df['aa_seq'] = df['aa']
            temp_df['len'] = [len(seq) for seq in df['aa']]
            temp_df['antibody'] = df['Target']
            temp_df['label'] = df['Label']

            data = pd.concat([data, temp_df], ignore_index=True)

    data = data.groupby(['aa_seq', 'antibody']).filter(lambda x: x['label'].nunique() == 1)
    data.drop_duplicates(subset=['aa_seq', 'antibody'], keep='first', inplace=True)

    data.to_csv("rbd_data_full.csv", index=False)
