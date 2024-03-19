import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from plmfit.shared_utils import utils

"""
Already done:

# Get rid of duplicate sequences with arbitrary labels
mouse = mouse[mouse.groupby('aa_sequence')['binding'].transform('nunique') == 1].copy(deep=True)
cattle = cattle[cattle.groupby('aa_sequence')['binding'].transform('nunique') == 1].copy(deep=True)
bat = bat[bat.groupby('aa_sequence')['binding'].transform('nunique') == 1].copy(deep=True)

# Drop sequences with ED greater than 8
    df.drop(df[df["ED"] > 8].index, inplace= True)
"""

if __name__=='__main__':
    
    # Imports the dataset
    data = utils.load_dataset("rbd")

    # We only keep the columns of interest to data analysis
    data = data[["aa_seq","len","ed","mouse","cattle","ihbat"]]

    # Rename ihbat to bat
    data.rename({"ihbat":"bat"}, axis = 1, inplace = True)

    # Convert bind and non to 1 and 0
    for species in ["mouse","cattle","bat"]:
        data.loc[data[species] == "bind",species] = 1
        data.loc[data[species] == "non",species] = 0
    
    # Dropping duplicates just in case there are any
    data.drop_duplicates(subset="aa_seq", keep="first", inplace=True)

    # Training and test splits are created
    data["random"] = "train"
    n_test = 1000
    test_ind = data.dropna().sample(n_test,replace = False).index
    data.loc[test_ind,"random"] = "test"

    # Validation split is created
    n_train = data[data["random"] == "train"].shape[0]
    val_ind = data[data["random"] == "train"].sample(round(n_train*0.1),replace = False).index
    data.loc[val_ind,"random"] = "validation"
    
    # Export to csv
    data.to_csv("plmfit/data/rbd/rbd_data_full.csv",index = False)