import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json

def main():
    # Define the index layers mapping
    index_layers = {0: 0, 1: 25, 2: 50, 3: 75, 4: 100}

    with open("./results/results_matrices.json", "r") as file:
        results_json = json.load(file)
        aav_sampled = results_json["AAV sampled"]
        aav_one_vs_rest = results_json["AAV one-vs-rest"]
        gb1_three_vs_rest = results_json["GB1 three-vs-rest"]
        gb1_one_vs_rest = results_json["GB1 one-vs-rest"]
        meltome_mixed = results_json["Meltome mixed"]


    colors = {
        'ProteinBERT': '#FF8C00',
        'ProGen2-small': '#98FB98',
        'ProGen2-medium': '#9ACD32',
        'ProGen2-xlarge': '#006400',
        'ESM2-650M':'#87CEFA',
        'ESM2-3B':'#4169E1',
        'ESM2-15B':'navy'
    }

    markers = {
        'ProteinBERT': '*',
        'ProGen2-small': '*',
        'ProGen2-medium': 'X',
        'ProGen2-xlarge': 'D',
        'ESM2-650M': '*',
        'ESM2-3B': 'X',
        'ESM2-15B': 'D'
    }


    # Define datasets, baselines, and dict_names
    datasets_fe = [
        aav_sampled["feature_extraction"],
        gb1_three_vs_rest["feature_extraction"],
        meltome_mixed["feature_extraction"],
    ]

    datasets_lora = [
        aav_sampled["lora"],
        gb1_three_vs_rest["lora"],
        meltome_mixed["lora"],
    ]

    datasets_adapters = [
        aav_sampled["adapters"],
        gb1_three_vs_rest["adapters"],
        meltome_mixed["adapters"],
    ]

    baselines = [
        aav_sampled["ohe_baseline"],
        gb1_three_vs_rest["ohe_baseline"],
        meltome_mixed["ohe_baseline"],
    ]


    dict_names = [
        'AAV-sampled',
        'GB1-three vs rest',
        'Meltome-mixed'
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
    # Function to filter out extrapolation and reduce marker size when y == 0.2
    def process_plot(numeric_x, numeric_y, ax, model, colors, markers):
        # Split the data into two parts: one with y != 0.2 and the other with y == 0.2
        small_marker_x = [x for x, y in zip(numeric_x, numeric_y) if y == 0.2]
        small_marker_y = [y for y in numeric_y if y == 0.2]
        
        full_marker_x = [x for x, y in zip(numeric_x, numeric_y) if y != 0.2]
        full_marker_y = [y for y in numeric_y if y != 0.2]
        
        # Plot points with y != 0.2 using normal line and markers
        if full_marker_x and full_marker_y:
            sns.lineplot(x=full_marker_x, y=full_marker_y, label=model, color=colors[model], 
                        marker=markers[model], markersize=20, ax=ax, linewidth=4)
        
        # Plot points where y == 0.2 using very small markers and no line
        if small_marker_x and small_marker_y:
            sns.scatterplot(x=small_marker_x, y=small_marker_y, label=model, color=colors[model], 
                            marker=markers[model], s=10, ax=ax)  # 's' controls marker size in scatterplot

    # Plot Feature Extraction (Row 1)
    for idx, (data, ax, baseline, name) in enumerate(zip(datasets_fe, axes[0, :], baselines, dict_names)):
        y_values_matrix = []
        numeric_x = convert_to_numeric(list(index_layers.values()))
        
        for model, values in data.items():
            numeric_y = convert_to_numeric(values)

            # Process plotting with reduced marker size and no extrapolation for y == 0.2
            process_plot(numeric_x, numeric_y, ax, model, colors, markers)

            # Create custom handle for this line if not already added
            if model not in handles_dict:
                handles_dict[model] = Line2D([0], [0], color=colors[model], marker=markers[model], markersize=20, linewidth=4, label=model)
            
            # Store the y-values to compute the min/max later
            y_values_matrix.append(numeric_y)
        
        # Compute the min and max y-values for each x-value
        y_values_matrix = np.array(y_values_matrix)
        min_y_values = np.min(y_values_matrix, axis=0)
        max_y_values = np.max(y_values_matrix, axis=0)

        # Plot baseline as a dashed line with thicker line
        baseline_handle = Line2D([0], [0], color='red', linestyle='--', linewidth=4, label='OHE - baseline')
        ax.axhline(y=baseline, color='red', linestyle='--', label='OHE - baseline', linewidth=4)
        
        # Set y-axis limits and customize ticks
        ax.set_ylim(0.3, 1)  # Adjust y-axis limits
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

            # Process plotting with reduced marker size and no extrapolation for y == 0.2
            process_plot(numeric_x, numeric_y, ax, model, colors, markers)

            # Create custom handle for this line if not already added
            if model not in handles_dict:
                handles_dict[model] = Line2D([0], [0], color=colors[model], marker=markers[model], markersize=20, linewidth=4, label=model)
            
            # Store the y-values to compute the min/max later
            y_values_matrix.append(numeric_y)
        
        # Compute the min and max y-values for each x-value
        y_values_matrix = np.array(y_values_matrix)
        min_y_values = np.min(y_values_matrix, axis=0)
        max_y_values = np.max(y_values_matrix, axis=0)

        # Plot baseline as a dashed line with thicker line
        baseline_handle = Line2D([0], [0], color='red', linestyle='--', linewidth=4, label='OHE - baseline')
        ax.axhline(y=baseline, color='red', linestyle='--', label='OHE - baseline', linewidth=4)
        
        # Set y-axis limits and customize ticks
        ax.set_ylim(0.3, 1)  # Adjust y-axis limits
        ax.set_yticks(get_y_ticks(ax))  # Set custom y-ticks
        
        ax.set_title(f'{name}', fontsize=35)
        ax.set_xlabel('')
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='both', labelsize=35)
        ax.tick_params(axis='y', which='both', labelsize=35)
        ax.set_ylabel('LoRA - Performance', fontsize=35)
        
        # Remove individual legends from subplots
        ax.legend_.remove()

    for idx, (data, ax, baseline, name) in enumerate(zip(datasets_adapters, axes[2, :], baselines, dict_names)):
        y_values_matrix = []
        numeric_x = convert_to_numeric(list(index_layers.values()))
        
        for model, values in data.items():
            numeric_y = convert_to_numeric(values)

            # Process plotting with reduced marker size and no extrapolation for y == 0.2
            process_plot(numeric_x, numeric_y, ax, model, colors, markers)

            # Create custom handle for this line if not already added
            if model not in handles_dict:
                handles_dict[model] = Line2D([0], [0], color=colors[model], marker=markers[model], markersize=20, linewidth=4, label=model)
            
            # Store the y-values to compute the min/max later
            y_values_matrix.append(numeric_y)
        
        # Compute the min and max y-values for each x-value
        y_values_matrix = np.array(y_values_matrix)
        min_y_values = np.min(y_values_matrix, axis=0)
        max_y_values = np.max(y_values_matrix, axis=0)

        # Plot baseline as a dashed line with thicker line
        baseline_handle = Line2D([0], [0], color='red', linestyle='--', linewidth=4, label='OHE - baseline')
        ax.axhline(y=baseline, color='red', linestyle='--', label='OHE - baseline', linewidth=4)

        # Add baseline handle to the dictionary
        handles_dict['OHE - baseline'] = baseline_handle  # Add this line

        # Set y-axis limits and customize ticks
        ax.set_ylim(0.3, 1)  # Adjust y-axis limits
        ax.set_yticks(get_y_ticks(ax))  # Set custom y-ticks
        
        ax.set_title(f'{name}', fontsize=35)
        ax.set_xlabel('')
        ax.set_xticks([0, 25, 50, 75, 100])  # Set x-axis ticks
        ax.set_xticklabels(['0', '25', '50', '75', '100'], fontsize=30)  # Set x-axis tick labels
        ax.tick_params(axis='x', which='both', labelsize=35)
        ax.tick_params(axis='y', which='both', labelsize=35)
        ax.set_ylabel('Adapters - Performance', fontsize=35)
        
        # Remove individual legends from subplots
        ax.legend_.remove()

    # Set common x-axis label
    fig.text(0.5, -0.01, '% of Layers used', ha='center', va='center', fontsize=35)

    # Add a single legend for all subplots at the bottom of the figure
    fig.legend(handles=list(handles_dict.values()), labels=list(handles_dict.keys()), loc='lower center', 
            bbox_to_anchor=(0.5, -0.06), ncol=len(handles_dict), frameon=False, fontsize=35)

    plt.tight_layout()
    plt.savefig('results/all_layer_analysis_with_lightgray_shaded_area_single_legend.png', bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    main()