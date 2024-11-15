import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import results_visualization.results_matrices as results_matrices

gb1 = results_matrices.ed_gb1_one_vs_rest
aav = results_matrices.ed_aav_one_vs_rest 
herh3 = results_matrices.ed_herh3_one_vs_rest 
rbd = results_matrices.ed_rbd_one_vs_rest

# Function to plot data
def plot_dict(ax, data_dict, title, legend_labels, show_yticks=True):
    for key, value in data_dict.items():
        if isinstance(value, list):
            # Determine the number of x-values
            no_x = len(value)
            if no_x > 7:
                no_x = 7
            x_values = np.arange(2, 2 + no_x)
            # Prepare x-values and y-values
            extended_x_values = np.arange(2, 2 + no_x)  # x-values from 2 to no_x + 1
            extended_y_values = value[:no_x]  # use only up to no_x values
            
            # Color and alpha settings
            color = 'blue'
            alpha = 0.05
            if key == 'OHE - baseline':
                color = 'red'
                alpha = 0.15
            elif key == 'LoRA':
                color = 'orange'
            elif key == 'Adapters':
                color = 'green'
            elif key == 'FE':
                color = 'blue'
                alpha = 0.1

            # Plot the line with markers
            line, = ax.plot(extended_x_values, extended_y_values, label=key, color=color)
            
            # Fill the area below the line with the same color
            ax.fill_between(extended_x_values, extended_y_values, color=color, alpha=alpha)
            
            if key not in legend_labels:
                legend_labels[key] = line
    
    ax.set_title(title, fontsize=20)  # Increase title font size
    ax.set_xticks(range(2, no_x + 2))
    ax.set_xticklabels(range(2, no_x + 2), fontsize=20)  # Increase x-tick font size

    # Set y-ticks to desired values or remove them
    if show_yticks:
        ax.set_yticks(np.arange(0, 1.1, 0.2))
    else:
        ax.set_yticks([])  # Hide y-ticks for this subplot
    ax.tick_params(axis='y', labelsize=20)  # Increase y-tick font size

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(25, 15))

# Replace these with your actual data
data_dicts = [aav, rbd, gb1, herh3]  # List of your dictionaries

# Initialize legend labels dictionary
legend_labels = {}

# Plot each dictionary

titles = [  'AAV-one vs rest', 'RBD-one vs rest', 'GB1-one vs rest', 'Trastuzumab-one vs rest']
for i, (ax, data_dict, title) in enumerate(zip(axs.flatten(), data_dicts, titles)):
    # Show y-ticks only in the left column (i % 2 == 0)
    plot_dict(ax, data_dict, title, legend_labels, show_yticks=(i % 2 == 0))

# Create a single legend for all subplots
# Move legend to the right side of the figure
fig.legend(handles=legend_labels.values(), labels=legend_labels.keys(), loc='center left', 
           bbox_to_anchor=(0.66, 0.4), frameon=False, fontsize=20)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(left=0.15, right=0.8, wspace=0.1)  # Decrease wspace to bring columns closer

# Add a single x-axis label for the entire figure
fig.text(0.46, -0.01, 'Edit Distance', ha='center', va='center', fontsize=25)

# Add a single y-axis label for the entire figure
fig.text(0.11, 0.5, 'Performance', ha='center', va='center', rotation='vertical', fontsize=25)

# Save the plot
plt.savefig('results_visualization/ed_plot.png', dpi=300)
