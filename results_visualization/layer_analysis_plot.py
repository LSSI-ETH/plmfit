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
    'AAV - sampled',
    'AAV - one_vs_rest',
    'GB1 - three_vs_rest',
    'GB1 - one_vs_rest',
    'Meltome - mixed'
]


fig1, axes1 = plt.subplots(nrows=1, ncols=5, figsize=(25, 5), sharey=True)

# Plot each dataset for Feature Extraction
for idx, (data, ax, baseline, name) in enumerate(zip(datasets_fe, axes1.flatten(), baselines, dict_names)):
    for model, values in data.items():
        sns.lineplot(x=list(index_layers.values()), y=values, label=model, color=colors[model], marker=markers[model], markersize=10, ax=ax)

    # Plot baseline as green dotted line
    ax.axhline(y=baseline, color='green', linestyle='--', label='Baseline')

    ax.set_title(f'{name}',fontsize = 22)
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.tick_params(axis='x', which='both', labelsize=17)
    ax.tick_params(axis='y', which='both', labelsize=17)
    ax.set_ylabel('Feature extracton - Performance',fontsize = 18)
    ax.legend().remove()  # Remove individual legend from subplot

# Set common x-axis label for Feature Extraction plots



plt.tight_layout()
plt.savefig('results_visualization/fe_layer_analysis.png',bbox_inches='tight', dpi=300)

# Create subplots for LORA
fig2, axes2 = plt.subplots(nrows=1, ncols=5, figsize=(25, 5), sharey=True)

# Plot each dataset for LORA
for idx, (data, ax, baseline, name) in enumerate(zip(datasets_lora, axes2.flatten(), baselines, dict_names)):
    for model, values in data.items():
        sns.lineplot(x=list(index_layers.values()), y=values, label=model, color=colors[model], marker=markers[model], markersize=10, ax=ax)

    # Plot baseline as green dotted line
    ax.axhline(y=baseline, color='green', linestyle='--', label='OHE - baseline')


    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.tick_params(axis='x', which='both', labelsize=17)
    ax.tick_params(axis='y', which='both', labelsize=17)
    ax.set_ylabel('LoRA - Performance',fontsize = 20)
    ax.legend().remove()  



plt.tight_layout()
plt.savefig('results_visualization/lora_layer_analysis.png',bbox_inches='tight', dpi=300)

# Create subplots for Adapters
fig3, axes3 = plt.subplots(nrows=1, ncols=5, figsize=(25, 5), sharey=True)

# Plot each dataset for Adapters
for idx, (data, ax, baseline, name) in enumerate(zip(datasets_adapters, axes3.flatten(), baselines, dict_names)):
    for model, values in data.items():
        sns.lineplot(x=list(index_layers.values()), y=values, label=model, color=colors[model], marker=markers[model], markersize=10, ax=ax)

    # Plot baseline as green dotted line
    ax.axhline(y=baseline, color='green', linestyle='--', label='Baseline')

    ax.tick_params(axis='x', which='both', labelsize=17)
    ax.tick_params(axis='y', which='both', labelsize=17)
    ax.set_xlabel('')
    ax.set_ylabel('Adapters - Performance',fontsize = 20)
    ax.legend().remove()  


# Set common x-axis label for Adapters plots
fig3.text(0.5, -0.02, ' % of Layers used', ha='center', va='center', fontsize = 14)

handles, labels = axes3[0].get_legend_handles_labels()
fig3.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.17), ncol=len(labels),  frameon=False,fontsize='16')


plt.tight_layout()
plt.savefig('results_visualization/adapters_layer_analysis.png',bbox_inches='tight', dpi=300)
