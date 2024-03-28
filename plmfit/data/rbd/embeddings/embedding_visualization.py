import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

reduction_type = "PCA" # Can be turned into a parser later
color_by = "antibody" 

# Step 1: Load the CSV Data
csv_file = "/cluster/home/srenwick/plmfit/plmfit/data/rbd/rbd_data_full.csv"  # Update with your file path
data = pd.read_csv(csv_file)
current_datetime = datetime.now().strftime('%Y-%m-%d_%H:%M')

# Step 2: Extract Features and Labels
sequences = data['aa_seq'].values
labels = data['label'].values
antibodies = data['antibody'].values

rbd_embeddings = torch.load("/cluster/scratch/srenwick/extract_embeddings/rbd_progen2-small_embs_middle_mean/rbd_progen2-small_embs_middle_mean.pt", map_location=torch.device('cpu'))

# Convert PyTorch tensor to NumPy array
embedding_numpy = rbd_embeddings.detach().cpu().numpy()

if reduction_type == "UMAP":
    # Step 4: UMAP Dimensionality Reduction
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.75, metric='euclidean', spread=1.0, random_state=42)
    embedding_umap = reducer.fit_transform(embedding_numpy)

    # Step 5: Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding_umap[:, 0], embedding_umap[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
    plt.colorbar(label='Binding Information')
    plt.title('UMAP Visualization of Progen Embeddings')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.savefig(f"/cluster/home/srenwick/plmfit/plmfit/data/rbd/UMAP_Visualization_{current_datetime}.png")

elif reduction_type == "PCA":
    # Step 4: PCA Dimensionality Reduction
    pca = PCA(n_components=10, random_state=42)
    embedding_pca = pca.fit_transform(embedding_numpy)

    # Step 5: Plotting
    plt.figure(figsize=(10, 8))
    if color_by == "antibody":
        for antibody in np.unique(antibodies):
            indices = np.where(antibodies == antibody)
            plt.scatter(embedding_pca[indices, 0], embedding_pca[indices, 1], label=antibody, alpha=0.7)
            plt.legend()
    else:
        plt.scatter(embedding_pca[:, 0], embedding_pca[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
        plt.colorbar(label='Binding Information')
    plt.title('PCA Visualization of Progen Embeddings')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(f"/cluster/home/srenwick/plmfit/plmfit/data/rbd/PCA_Visualization_{current_datetime}.png")

elif reduction_type == "TSNE":
    # Step 4: t-SNE Dimensionality Reduction
    tsne = TSNE(n_components=2, random_state=42)
    embedding_tsne = tsne.fit_transform(embedding_numpy)

    # Step 5: Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding_tsne[:, 0], embedding_tsne[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
    plt.colorbar(label='Binding Information')
    plt.title('t-SNE Visualization of Progen Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(f"/cluster/home/srenwick/plmfit/plmfit/data/rbd/tSNE_Visualization_{current_datetime}.png")

else:
    raise ValueError('Unsupported reduction option')