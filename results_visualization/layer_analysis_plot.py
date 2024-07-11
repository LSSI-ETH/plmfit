import seaborn as sns
import matplotlib.pyplot as plt
import results_visualization.results_matrices as results_matrices

# Define the index layers mapping
index_layers =results_matrices.index_layers 

# Extract data for each category
feature_extraction_data = results_matrices.aav_sampled_dict['feature_extraction']
lora_data = results_matrices.aav_sampled_dict['lora']
adapters_data = results_matrices.aav_sampled_dict['adapters']



colors = {
    'ProteinBERT': 'blue',
    'ProGen2-small': 'orange',
    'ProGen2-medium': 'orange',
    'ProGen2-xlarge': 'orange'
}

markers = {
    'ProteinBERT': '*',
    'ProGen2-small': '*',
    'ProGen2-medium': 'X',
    'ProGen2-xlarge': 'D'
}


# Define datasets, baselines, and dict_names
datasets_fe = [
    results_matrices.aav_sampled_dict['feature_extraction'],
    results_matrices.aav_one_vs_rest_dict['feature_extraction'],
    results_matrices.gb1_three_vs_rest_dict['feature_extraction'],
    results_matrices.gb1_one_vs_rest_dict['feature_extraction'],
    results_matrices.meltome_mixed_dict['feature_extraction']
]

datasets_lora = [
    results_matrices.aav_sampled_dict['lora'],
    results_matrices.aav_one_vs_rest_dict['lora'],
    results_matrices.gb1_three_vs_rest_dict['lora'],
    results_matrices.gb1_one_vs_rest_dict['lora'],
    results_matrices.meltome_mixed_dict['lora']
]

datasets_adapters = [
    results_matrices.aav_sampled_dict['adapters'],
    results_matrices.aav_one_vs_rest_dict['adapters'],
    results_matrices.gb1_three_vs_rest_dict['adapters'],
    results_matrices.gb1_one_vs_rest_dict['adapters'],
    results_matrices.meltome_mixed_dict['adapters']
]

baselines = [
    results_matrices.aav_sampled_dict['ohe_baseline'],
    results_matrices.aav_one_vs_rest_dict['ohe_baseline'],
    results_matrices.gb1_three_vs_rest_dict['ohe_baseline'],
    results_matrices.gb1_one_vs_rest_dict['ohe_baseline'],
    results_matrices.meltome_mixed_dict['ohe_baseline']
]

dict_names = [
    'aav_sampled_dict',
    'aav_one_vs_rest_dict',
    'gb1_three_vs_rest_dict',
    'gb1_one_vs_rest_dict',
    'meltome_mixed_dict'
]


fig1, axes1 = plt.subplots(nrows=1, ncols=5, figsize=(25, 5), sharey=True)

# Plot each dataset for Feature Extraction
for idx, (data, ax, baseline, name) in enumerate(zip(datasets_fe, axes1.flatten(), baselines, dict_names)):
    for model, values in data.items():
        sns.lineplot(x=list(index_layers.values()), y=values, label=model, color=colors[model], marker=markers[model], markersize=10, ax=ax)

    # Plot baseline as green dotted line
    ax.axhline(y=baseline, color='green', linestyle='--', label='Baseline')

    ax.set_title(f'{name.capitalize()} - Feature Extraction ')
    ax.set_xlabel('')
    ax.set_ylabel('Performance Metric')
    ax.legend().remove()  # Remove individual legend from subplot

# Set common x-axis label for Feature Extraction plots
fig1.text(0.5, 0, 'Layers Used', ha='center', va='center')

# Create a single legend below the subplots
handles, labels = axes1[0].get_legend_handles_labels()
# Filter out the dictionary name from labels
filtered_labels = [label.split(' - ')[0] if 'Baseline' not in label else label for label in labels]
fig1.legend(handles, filtered_labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=len(filtered_labels), frameon=False)

plt.tight_layout()
plt.show()

# Create subplots for LORA
fig2, axes2 = plt.subplots(nrows=1, ncols=5, figsize=(25, 5), sharey=True)

# Plot each dataset for LORA
for idx, (data, ax, baseline, name) in enumerate(zip(datasets_lora, axes2.flatten(), baselines, dict_names)):
    for model, values in data.items():
        sns.lineplot(x=list(index_layers.values()), y=values, label=model, color=colors[model], marker=markers[model], markersize=10, ax=ax)

    # Plot baseline as green dotted line
    ax.axhline(y=baseline, color='green', linestyle='--', label='OHE - baseline')

    ax.set_title(f'{name.capitalize()} - LoRA ')
    ax.set_xlabel('')
    ax.set_ylabel('Performance Metric')
    ax.legend().remove()  

# Set common x-axis label for LORA plots
fig2.text(0.5, 0, 'Layers Used', ha='center', va='center')

# Create a single legend for LORA plots
handles, labels = axes2[0].get_legend_handles_labels()
fig2.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=len(labels),  frameon=False)


plt.tight_layout()
plt.show()

# Create subplots for Adapters
fig3, axes3 = plt.subplots(nrows=1, ncols=5, figsize=(25, 5), sharey=True)

# Plot each dataset for Adapters
for idx, (data, ax, baseline, name) in enumerate(zip(datasets_adapters, axes3.flatten(), baselines, dict_names)):
    for model, values in data.items():
        sns.lineplot(x=list(index_layers.values()), y=values, label=model, color=colors[model], marker=markers[model], markersize=10, ax=ax)

    # Plot baseline as green dotted line
    ax.axhline(y=baseline, color='green', linestyle='--', label='Baseline')

    ax.set_title(f'{name.capitalize()} - Adapters ')
    ax.set_xlabel('')
    ax.set_ylabel('Performance Metric')
    ax.legend().remove()  

# Set common x-axis label for Adapters plots
fig3.text(0.5, 0, 'Layers Used', ha='center', va='center')

handles, labels = axes3[0].get_legend_handles_labels()
fig3.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=len(labels),  frameon=False)


plt.tight_layout()
plt.show()
