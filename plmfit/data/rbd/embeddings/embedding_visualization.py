import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import torch

# Step 1: Load the CSV Data
csv_file = "plmfit/plmfit/data/rbd/rbd_data_full.csv"  # Update with your file path
data = pd.read_csv(csv_file)

# Step 2: Extract Features and Labels
sequences = data['aa.seq'].values
labels = data['label'].values

rbd_embeddings = torch.load("plmfit/plmfit/data/rbd/embeddings/rbd_esm2_t33_650M_UR50D_embs_layer17_mean.pt", map_location=torch.device('cpu'))

# Convert PyTorch tensor to NumPy array
embedding_numpy = rbd_embeddings.detach().cpu().numpy()

# Step 4: UMAP Dimensionality Reduction
reducer = umap.UMAP()
embedding_umap = reducer.fit_transform(embedding_numpy)

# Step 5: Plotting
plt.figure(figsize=(10, 8))
plt.scatter(embedding_umap[:, 0], embedding_umap[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
plt.colorbar(label='Binding Information')
plt.title('UMAP Visualization of Progen Embeddings')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.show()