import results_visualization.results_matrices as results_matrices
import matplotlib.pyplot as plt
import numpy as np

# Index labeling
index_tl_techniques = {0: 'FE', 1: 'LoRA', 2: 'LoRA-', 3: 'Adapters', 4: 'Adapters-'}

# Colors and markers for each model
color_marker_dict = {
    'ProteinBERT': ('blue', '*'),
    'ProGen2-small': ('orange', '*'),
    'ProGen2-medium': ('orange', 'X'),
    'ProGen2-xlarge': ('orange', 'D')
}

# Example datasets (replace these with actual data from results_matrices)
datasets = [
    results_matrices.aav_sampled_dict['best_models'],
    results_matrices.aav_one_vs_rest_dict['best_models'],
    results_matrices.gb1_three_vs_rest_dict['best_models'],
    results_matrices.gb1_one_vs_rest_dict['best_models'],
    results_matrices.meltome_mixed_dict['best_models']
]

# Baselines for each dataset
baselines = [
    results_matrices.aav_sampled_dict['ohe_baseline'],
    results_matrices.aav_one_vs_rest_dict['ohe_baseline'],
    results_matrices.gb1_three_vs_rest_dict['ohe_baseline'],
    results_matrices.gb1_one_vs_rest_dict['ohe_baseline'],
    results_matrices.meltome_mixed_dict['ohe_baseline']
]

# Names for each dataset
dict_names = [
    'aav_sampled_dict',
    'aav_one_vs_rest_dict',
    'gb1_three_vs_rest_dict',
    'gb1_one_vs_rest_dict',
    'meltome_mixed_dict'
]

# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 5), sharey=True)

# Plot each dataset
for ax, data, baseline, name in zip(axes, datasets, baselines, dict_names):
    for i in range(len(index_tl_techniques)):
        color = 'lightgray' if i % 2 == 0 else 'white'
        ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.3)
    
    for model_name, values in data.items():
        x = np.arange(len(values))  # x-axis: position in the list (0-indexed)
        noise = np.random.uniform(-0.3, 0.3, len(values))  # Adding small noise to x positions
        x_noise = x + noise  # Adding noise to x positions
        color, marker = color_marker_dict[model_name]
        ax.scatter(x_noise, values, label=model_name, color=color, marker=marker, s=100)  # Increased marker size
    
    # Plot the baseline
    ax.axhline(y=baseline, color='green', linestyle='--', label='Baseline')
    
    ax.set_xticks(range(len(index_tl_techniques)))
    ax.set_xticklabels([index_tl_techniques[i] for i in range(len(index_tl_techniques))])
    ax.set_xlabel('Techniques')
    ax.grid(True)
    ax.set_title(name.replace('_', ' ').capitalize())

# Add a single legend below the subplots
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=len(labels), fontsize='large', frameon=False)

# Set common y-axis label and title
fig.text(0.5, 0.04, 'Techniques', ha='center', va='center')
fig.text(0.04, 0.5, 'Values', ha='center', va='center', rotation='vertical')
fig.suptitle('Scatter Plot of Protein Language Models Performance with Noise', y=1.02)

plt.tight_layout()
plt.show()


