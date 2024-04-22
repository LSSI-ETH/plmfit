import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

"""
data_parse.py prepares omicronab_data_full.csv for use with plmfit
- enters "./raw_data/raw_data" and searches for the relevant files (correct antibodies & cleaner data)
- cleans data, removes redundant and contradictory sequences
- creates rbd_data_full.csv
"""

def parse_data(file_path):
    # Read the CSV file into a DataFrame
    data_full = pd.read_csv(file_path)
    
    # Concatenate the V_h and V_l columns to create aa_seq
    data_full['aa_seq'] = data_full['V_h']
    
    # Calculate the length of aa_seq and store it in the len column
    data_full['len'] = data_full['aa_seq'].apply(len)
    
    # Rename the Bind column to label and the ID column to antibody
    data_full = data_full.rename(columns={'Bind': 'label', 'ID': 'antibody'})
    
    # Select only the desired columns
    data_parsed = data_full[['aa_seq', 'len', 'label', 'antibody']]
    
    # Save the parsed DataFrame to a new CSV file
    output_file = 'omicronab_data_full.csv'
    data_parsed.to_csv(output_file, index=False)
    
    #print(f"Data parsed and saved to {output_file}")


if __name__ == '__main__':

    # Define the file path
    file_path = 'data_full.csv'

    # Call the parse_data function with the file path
    parse_data(file_path)