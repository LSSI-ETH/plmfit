import results_visualization.results_matrices as results_matrices
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    ax.grid(True)
    ax.set_title(name.replace('_', ' ').capitalize())

# Remove x-labels for each subplot
for ax in axes:
    ax.set_xlabel('')

# Add a single legend below the subplots
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=len(labels), fontsize='large', frameon=False)

# Set common y-axis label
fig.text(0, 0.5, 'Evaluation', ha='center', va='center', rotation='vertical')

plt.tight_layout()
plt.show()



# Custom colors for legend and bars
custom_colors = {
    'Baseline': 'green',
    'FE': 'gray',
    'LoRA': 'brown',
    'LoRA-': 'lightcoral',
    'Adapters': 'peru',
    'Adapters-': 'peachpuff'
}
# Plotting
fig, axes = plt.subplots(nrows=1, ncols=len(datasets), figsize=(20, 5), sharey=True)

for ax, data, baseline, dict_name in zip(axes, datasets, baselines, dict_names):
    # Extracting data into a DataFrame
    plot_data = []
    models = list(data.keys())
    for model in models:
        for i, value in enumerate(data[model]):
            plot_data.append([model, index_tl_techniques[i], value])

    df = pd.DataFrame(plot_data, columns=['Model', 'Technique', 'Value'])

    # Prepare data for plotting
    num_models = len(models)
    num_techniques = len(index_tl_techniques)
    bar_width = 0.2  # Width of each bar
    bar_positions = np.arange(num_models) * 1.5  # Positions for the bars, multiplied by 1.5 for separation

    # Plotting
    for i, technique in enumerate(index_tl_techniques.values()):
        # Calculate x positions for bars in the group
        x_positions = bar_positions + i * bar_width

        # Select data for current technique
        df_tech = df[df['Technique'] == technique]

        # Plot bars for current technique with custom colors
        ax.bar(x_positions, df_tech['Value'], width=bar_width, label=technique, alpha=0.7, color=custom_colors[technique])

    # Plotting the baseline (green line)
    ax.axhline(y=baseline, color='green', linestyle='--', label='Baseline')

    # Set x-axis labels for each group of barplots (Protein Language Models)
    ax.set_xticks(bar_positions + (num_techniques / 2 - 0.5) * bar_width)
    ax.set_xticklabels(models, rotation=45, ha='right')

    ax.set_xlabel('')

    ax.set_title(dict_name.replace('_', ' ').capitalize())

# Adding common y-axis label
axes[0].set_ylabel('Values')

# Combine legends into a single legend outside the subplots
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=num_techniques + 1, frameon=False)

# Adjust layout
plt.tight_layout()
plt.show()
