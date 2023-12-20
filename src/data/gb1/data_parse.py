import argparse
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import warnings
import seaborn as sns
from Bio import SeqIO
import Bio.Seq
from Bio.Seq import Seq
import os
import math

parser = argparse.ArgumentParser(description='gb1_parser')
parser.add_argument('--split', type=str, default='one_vs_rest') ## options ['progen2-small', 'progen2-xlarge', 'progen2-oas', 'progen2-medium', 'progen2-base', 'progen2-BFD90' , 'progen2-large']


args = parser.parse_args()


def load_data(split):
    return pd.read_csv('/Users/tbikias/workspace/dms_pLM/data/gb1/four_mutations_full_data.csv') #f'./splits/{split}.csv'

#

data = load_data(args.split)
wild_type = data[data['target'] == 1]['sequence'].values[0]
c = 0
def get_pos_mut(seq):
    ret = []
    for i in range(len(seq)):
       if seq[i] != wild_type[i]:
            ret.append(i)
            ret.append(seq[i])
    if len(ret) == 0: 
        ret = [-1 , '-']
    
    if len(ret) > 2:
        ret = [-2, '@']
    return ret

if __name__ == '__main__':
    
    data_format = pd.DataFrame(columns = ["aa_seq","pos_mut", "aa_new", "score" , "err"])
    data_format['aa_seq'] = data['sequence']
    data_format['score'] = data['target']
    data_format['set'] = data['set']
    data_format['pos_mut'] = data['sequence'].apply(lambda x : get_pos_mut(x)[0])
    data_format['aa_new'] = data['sequence'].apply(lambda x : get_pos_mut(x)[1])
    data_format['len'] = data_format.apply(lambda x : len(x))
    data_format.reset_index(inplace = True)
    data_format.to_csv('data_format.csv')
