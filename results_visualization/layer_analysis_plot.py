import seaborn as sns
import matplotlib.pyplot as plt
import results_visualization.results_matrices as results_matrices
import numpy as np
# Define the index layers mapping
index_layers =results_matrices.index_layers 

# Extract data for each category
feature_extraction_data = results_matrices.aav_sampled_dict['feature_extraction']
lora_data = results_matrices.aav_sampled_dict['lora']
adapters_data = results_matrices.aav_sampled_dict['adapters']



colors = {
    'ProteinBERT': 'green',
    'ProGen2-small': 'orange',
    'ProGen2-medium': 'orange',
    'ProGen2-xlarge': 'orange',
    'ESM-650M':'blue',
    'ESM-3B':'blue',
    'ESM-15B':'blue'
}

markers = {
    'ProteinBERT': '*',
    'ProGen2-small': '*',
    'ProGen2-medium': 'X',
    'ProGen2-xlarge': 'D',
    'ESM-650M': '*',
    'ESM-3B': 'X',
    'ESM-15B': 'D'
}


# Define datasets, baselines, and dict_names
datasets_fe = [
    results_matrices.aav_sampled_dict['feature_extraction'],
    results_matrices.gb1_three_vs_rest_dict['feature_extraction'],
    results_matrices.meltome_mixed_dict['feature_extraction']
]

datasets_lora = [
    results_matrices.aav_sampled_dict['lora'],
    results_matrices.gb1_three_vs_rest_dict['lora'],
    results_matrices.meltome_mixed_dict['lora']
]

datasets_adapters = [
    results_matrices.aav_sampled_dict['adapters'],
    results_matrices.gb1_three_vs_rest_dict['adapters'],
    results_matrices.meltome_mixed_dict['adapters']
]

baselines = [
    results_matrices.aav_sampled_dict['ohe_baseline'],
    results_matrices.gb1_three_vs_rest_dict['ohe_baseline'],
    results_matrices.meltome_mixed_dict['ohe_baseline']
]



dict_names = [
    'AAV - sampled',
    'GB1 - three_vs_rest',
    'Meltome - mixed'
]

from matplotlib.lines import Line2D

from matplotlib.ticker import MaxNLocator

# Function to convert data to numeric types
def convert_to_numeric(data):
    try:
        return np.array(data, dtype=float)  # Convert to float
    except ValueError:
        raise ValueError(f"Non-numeric data found: {data}")

# Function to get y-axis ticks excluding 0.0
def get_y_ticks(ax):
    ticks = np.linspace(0, 1, num=6)  # Adjust the number of ticks as needed
    ticks = [tick for tick in ticks if tick != 0.0]
    return ticks

# Set up the figure with 3 rows and 3 columns for the plots
fig, axes = plt.subplots(nrows=3, ncols=len(dict_names), figsize=(50, 30), sharey='row')

# Dictionary to hold handles and labels for the legend
handles_dict = {}

# Plot Feature Extraction (Row 1)
for idx, (data, ax, baseline, name) in enumerate(zip(datasets_fe, axes[0, :], baselines, dict_names)):
    y_values_matrix = []
    numeric_x = convert_to_numeric(list(index_layers.values()))
    
    for model, values in data.items():
        numeric_y = convert_to_numeric(values)

        # Plot the value lines for each protein language model with increased line thickness
        line = sns.lineplot(x=numeric_x, y=numeric_y, label=model, color=colors[model], marker=markers[model], 
                            markersize=20, ax=ax, linewidth=4)

        # Create custom handle for this line if not already added
        if model not in handles_dict:
            handles_dict[model] = Line2D([0], [0], color=colors[model], marker=markers[model], markersize=20, linewidth=4, label=model)
        
        # Store the y-values to compute the min/max later
        y_values_matrix.append(numeric_y)
    
    # Compute the min and max y-values for each x-value
    y_values_matrix = np.array(y_values_matrix)
    min_y_values = np.min(y_values_matrix, axis=0)
    max_y_values = np.max(y_values_matrix, axis=0)
    
    # Shade the area between the min and max values in light gray
    ax.fill_between(numeric_x, min_y_values, max_y_values, color='lightgray', alpha=0.3)

    # Plot baseline as a dashed line with thicker line
    baseline_handle = Line2D([0], [0], color='red', linestyle='--', linewidth=4,  label='OHE - baseline')
    ax.axhline(y=baseline, color='red', linestyle='--',  label='OHE - baseline', linewidth=4)
    
    # Add baseline handle to dictionary

    # Set y-axis limits and customize ticks
    ax.set_ylim(0, 1)
    ax.set_yticks(get_y_ticks(ax))  # Set custom y-ticks
    
    ax.set_title(f'{name}', fontsize=35)
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.tick_params(axis='x', which='both', labelsize=35)
    ax.tick_params(axis='y', which='both', labelsize=35)
    ax.set_ylabel('Feature extraction - Performance', fontsize=35)
    
    # Remove individual legends from subplots
    ax.legend_.remove()

# Plot LoRA (Row 2)
for idx, (data, ax, baseline, name) in enumerate(zip(datasets_lora, axes[1, :], baselines, dict_names)):
    y_values_matrix = []
    numeric_x = convert_to_numeric(list(index_layers.values()))
    
    for model, values in data.items():
        numeric_y = convert_to_numeric(values)

        # Plot the value lines for each protein language model with increased line thickness
        line = sns.lineplot(x=numeric_x, y=numeric_y, label=model, color=colors[model], marker=markers[model], 
                            markersize=20, ax=ax, linewidth=4)

        # Create custom handle for this line if not already added
        if model not in handles_dict:
            handles_dict[model] = Line2D([0], [0], color=colors[model], marker=markers[model], markersize=20, linewidth=4, label=model)
        
        # Store the y-values to compute the min/max later
        y_values_matrix.append(numeric_y)
    
    # Compute the min and max y-values for each x-value
    y_values_matrix = np.array(y_values_matrix)
    min_y_values = np.min(y_values_matrix, axis=0)
    max_y_values = np.max(y_values_matrix, axis=0)
    
    # Shade the area between the min and max values in light gray
    ax.fill_between(numeric_x, min_y_values, max_y_values, color='lightgray', alpha=0.3)

    # Plot baseline as a dashed line with thicker line
    baseline_handle = Line2D([0], [0], color='red', linestyle='--', linewidth=4, label='OHE - baseline')
    ax.axhline(y=baseline, color='red', linestyle='--', label='OHE - baseline', linewidth=4)
    
    # Add baseline handle to dictionary
    handles_dict['OHE - baseline'] = baseline_handle

    # Set y-axis limits and customize ticks
    ax.set_ylim(0, 1)
    ax.set_yticks(get_y_ticks(ax))  # Set custom y-ticks
    
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.tick_params(axis='x', which='both', labelsize=35)
    ax.tick_params(axis='y', which='both', labelsize=35)
    ax.set_ylabel('LoRA - Performance', fontsize=35)
    
    # Remove individual legends from subplots
    ax.legend_.remove()

# Plot Adapters (Row 3)
for idx, (data, ax, baseline, name) in enumerate(zip(datasets_adapters, axes[2, :], baselines, dict_names)):
    y_values_matrix = []
    numeric_x = convert_to_numeric(list(index_layers.values()))
    
    for model, values in data.items():
        numeric_y = convert_to_numeric(values)

        # Plot the value lines for each protein language model with increased line thickness
        line = sns.lineplot(x=numeric_x, y=numeric_y, label=model, color=colors[model], marker=markers[model], 
                            markersize=20, ax=ax, linewidth=4)

        # Create custom handle for this line if not already added
        if model not in handles_dict:
            handles_dict[model] = Line2D([0], [0], color=colors[model], marker=markers[model], markersize=20, linewidth=4, label=model)
        
        # Store the y-values to compute the min/max later
        y_values_matrix.append(numeric_y)
    
    # Compute the min and max y-values for each x-value
    y_values_matrix = np.array(y_values_matrix)
    min_y_values = np.min(y_values_matrix, axis=0)
    max_y_values = np.max(y_values_matrix, axis=0)
    
    # Shade the area between the min and max values in light gray
    ax.fill_between(numeric_x, min_y_values, max_y_values, color='lightgray', alpha=0.3)

    # Plot baseline as a dashed line with thicker line
    baseline_handle = Line2D([0], [0], color='red', linestyle='--', linewidth=4,  label='OHE - baseline')
    ax.axhline(y=baseline, color='red', linestyle='--',  label='OHE - baseline', linewidth=4)
    
    # Add baseline handle to dictionary

    # Set y-axis limits and customize ticks
    ax.set_ylim(0, 1)
    ax.set_yticks(get_y_ticks(ax))  # Set custom y-ticks

    ax.tick_params(axis='x', which='both', labelsize=35)
    ax.tick_params(axis='y', which='both', labelsize=35)
    ax.set_xlabel('')
    ax.set_ylabel('Adapters - Performance', fontsize=35)
    
    # Remove individual legends from subplots
    ax.legend_.remove()

# Set common x-axis label
fig.text(0.5, -0.01, '% of Layers used', ha='center', va='center', fontsize=35)

# Add a single legend for all subplots at the bottom of the figure
fig.legend(handles=list(handles_dict.values()), labels=list(handles_dict.keys()), loc='lower center', 
           bbox_to_anchor=(0.5, -0.06), ncol=len(handles_dict), frameon=False, fontsize=35)

plt.tight_layout()
plt.savefig('results_visualization/all_layer_analysis_with_lightgray_shaded_area_single_legend.png', bbox_inches='tight', dpi=300)


