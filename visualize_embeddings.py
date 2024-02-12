from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import torch

import pandas as pd

layers = ['last']
dataset = 'gb1'
model = 'progen2-small'
reduction = 'bos'

# Load dataset
dataset_path = f'./plmfit/data/{dataset}/{dataset}_data_full.csv'
output_path = f'./plmfit/data/{dataset}/embeddings/plots'
data = pd.read_csv(dataset_path)

# Ensure that the 'score' column aligns with your embeddings
scores = data['score'].values


scaler = MinMaxScaler()
scores_scaled = scaler.fit_transform(scores.reshape(-1, 1)).flatten()

for layer in layers:
    # Load embeddings
    file_path = f'./plmfit/data/{dataset}/embeddings/{dataset}_{model}_embs_layer{layer}_{reduction}.pt'
    embeddings = torch.load(file_path, map_location=torch.device('cpu'))
    embeddings = embeddings.numpy() if embeddings.is_cuda else embeddings


    # Perform PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(
        reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=scores, cmap='viridis')
    plt.title(
        f"2D PCA of {dataset} Embeddings\nScore coloring - Layer {layer} - {reduction} - {model}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label='Score')
    # Save the figure to a file
    plt.savefig(
        f'{output_path}/PCA_{dataset}_{model}_Layer-{layer}_{reduction}.png', bbox_inches='tight')

    plt.close()