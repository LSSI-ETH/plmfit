from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import torch
import plmfit.shared_utils.data_explore as de
import pandas as pd
import plmfit.shared_utils.utils as utils

layers = ['first', 'middle', 'last']
data_type = 'meltome'
model = 'progen2-xlarge'
reduction = 'mean'

# Define human-related species to be consolidated
# human_species = ['HepG2', 'HAOEC', 'colon_cancer', 'HL60', 'HEK293T',
#                  'U937', 'Jurkat', 'HaCaT', 'K562', 'pTcells']

# data = utils.load_dataset(data_type=data_type)
# Extract species from the identifier and consolidate human-related species
# data['species'] = data['id'].apply(
#     lambda x: 'Homo_sapiens' if any(human in x for human in human_species) else '_'.join(
#         x.split('_')[1:3]) if len(x.split('_')) > 3 else '_'.join(x.split('_')[1:])
# )

de.PCA_2d(data_type=data_type, model=model, layers=layers, reduction=reduction, labels_col='score', scaled=False)