import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import numpy as np

layers = ['first', 'middle', 'last']
dataset = 'meltome'
model = 'progen2-xlarge'
reduction = 'mean'

# Load dataset
dataset_path = f'./plmfit/data/{dataset}/{dataset}_data_full.csv'
output_path = f'./plmfit/data/{dataset}/embeddings/plots'
data = pd.read_csv(dataset_path)

# Define human-related species to be consolidated
human_species = ['HepG2', 'HAOEC', 'colon_cancer', 'HL60', 'HEK293T',
                 'U937', 'Jurkat', 'HaCaT', 'K562', 'pTcells']

# Extract species from the identifier and consolidate human-related species
data['species'] = data['id'].apply(
    lambda x: 'Homo_sapiens' if any(human in x for human in human_species) else '_'.join(
        x.split('_')[1:3]) if len(x.split('_')) > 3 else '_'.join(x.split('_')[1:])
)

# Step 1: Identify indices of 'Homo_sapiens' entries
human_indices = data.index[data['species'] == 'Homo_sapiens'].tolist()

for layer in layers:

    # Load embeddings
    file_path = f'./plmfit/data/{dataset}/embeddings/{dataset}_{model}_embs_layer{layer}_{reduction}.pt'
    embeddings = torch.load(file_path, map_location=torch.device('cpu'))
    embeddings = embeddings.numpy() if embeddings.is_cuda else embeddings

    # Step 2: Filter out embeddings corresponding to 'Homo_sapiens' entries
    # Create a mask for all indices not in human_indices
    mask = np.ones(len(embeddings), dtype=bool)  # Initially, all True
    mask[human_indices] = False  # Set False for human_indices
    filtered_embeddings = embeddings[mask]

    # Step 3: Filter the dataset to exclude 'Homo_sapiens'
    filtered_data = data[data['species'] != 'Homo_sapiens']

    # Ensure the filtered dataset and filtered embeddings are now aligned
    # Proceed with PCA and plotting

    # Perform PCA on filtered embeddings
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(filtered_embeddings)

    # Mapping species to colors for the filtered dataset
    species_unique = filtered_data['species'].unique()
    species_to_int = {species: i for i, species in enumerate(species_unique)}
    species_colors = filtered_data['species'].map(species_to_int).values

    # Generate a discrete colormap
    num_species = len(species_unique)
    cmap = plt.get_cmap('tab20', num_species)  # Using 'tab20' colormap

    # Plot with discrete colors
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(
        reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=species_colors, cmap=cmap, alpha=0.7)
    plt.title(
        f"2D PCA of Filtered {dataset} Embeddings\nSpecies coloring (Excl. Homo Sapiens) - Layer {layer} - {reduction} - {model}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    # Create a color bar with tick marks and labels for each species
    cbar = plt.colorbar(scatter, ticks=range(num_species))
    cbar.set_ticklabels(species_unique)
    cbar.set_label('Species')

    # Save the figure to a file
    plt.savefig(
        f'{output_path}/PCA_{dataset}_{model}_Layer-{layer}_{reduction}.png', bbox_inches='tight')

    # Optionally, display the plot as well
    # plt.show()

    # Close the plot to free up memory
    plt.close()


# # Apply UMAP
# reducer = umap.UMAP(random_state=42)
# umap_embeddings = reducer.fit_transform(embeddings)

# # Plot UMAP results with discrete colors
# plt.figure(figsize=(10, 10))
# scatter = plt.scatter(
#     umap_embeddings[:, 0], umap_embeddings[:, 1], c=species_colors, cmap='tab20', alpha=0.7)
# plt.title("UMAP of Filtered Meltome Embeddings - Discrete Coloring")
# plt.xlabel("UMAP 1")
# plt.ylabel("UMAP 2")

# # Create a color bar with tick marks and labels for each species
# num_species = len(species_to_int)
# cbar = plt.colorbar(scatter, ticks=range(num_species))
# cbar.set_ticklabels(list(species_to_int.keys()))
# cbar.set_label('Species')

# plt.show()
