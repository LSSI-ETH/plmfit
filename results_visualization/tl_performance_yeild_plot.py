import results_visualization.results_matrices as results_matrices
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
datasets = [
    results_matrices.aav_sampled_dict['best_models'],
    results_matrices.aav_one_vs_rest_dict['best_models'],
    results_matrices.gb1_three_vs_rest_dict['best_models'],
    results_matrices.gb1_one_vs_rest_dict['best_models'],
    results_matrices.meltome_mixed_dict['best_models']
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

# Prepare the data for plotting
plot_data = []
for data_dict, baseline, dict_name in zip(datasets, baselines, dict_names):
    for model, values in data_dict.items():
        for value in values:
            percentage_diff = (value - baseline) / baseline * 100
            plot_data.append([dict_name, model, percentage_diff])

df = pd.DataFrame(plot_data, columns=['Task', 'Model', 'PercentageDiff'])

# Plotting
fig, ax = plt.subplots(figsize=(14, 6))

# Create violin plots for each task
positions = np.arange(len(dict_names))
legend_labels = []

for i, task in enumerate(dict_names):
    task_data = df[df['Task'] == task]['PercentageDiff']
    
    # Splitting data for positive and negative regions
    positive_data = task_data[task_data >= 0]
    negative_data = task_data[task_data < 0]
    
    # Check if data is not empty before plotting
    if not positive_data.empty:
        parts_positive = ax.violinplot(positive_data, positions=[i], showmeans=False, showmedians=True)
        for pc in parts_positive['bodies']:
            pc.set_facecolor('green')
            pc.set_edgecolor('black')
            pc.set_linewidth(1)  # Set edge linewidth to 1
            pc.set_alpha(0.7)

        # Annotate maximum value above violin plot
        max_value = np.max(positive_data)
        ax.text(i, max_value + 0.5, f'{max_value:.2f}%', ha='center', va='bottom', color='green', fontsize=10)
        
        legend_labels.append('Increase')
    
    if not negative_data.empty:
        parts_negative = ax.violinplot(negative_data, positions=[i], showmeans=False, showmedians=True)
        for pc in parts_negative['bodies']:
            pc.set_facecolor('red')
            pc.set_edgecolor('black')
            pc.set_linewidth(1)  # Set edge linewidth to 1
            pc.set_alpha(0.7)
            
        # Annotate minimum value below violin plot
        min_value = np.min(negative_data)
        ax.text(i, min_value - 0.9, f'{min_value:.2f}%', ha='center', va='top', color='red', fontsize=10)
        
        legend_labels.append('Decrease')
    
    # Customizing lines (mean, median, max, min)
    for partname in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
        if not positive_data.empty:
            vp = parts_positive[partname]
            vp.set_edgecolor('black')  # Set edge color to black
            vp.set_linewidth(0.5)  # Set linewidth to 0.5
        if not negative_data.empty:
            vn = parts_negative[partname]
            vn.set_edgecolor('black')  # Set edge color to black
            vn.set_linewidth(0.5)  # Set linewidth to 0.5
            
# Set labels and title
#ax.set_xlabel('Tasks')
ax.set_ylabel('% difference in performance',fontsize=15)
ax.set_title(' Per task TL performance yield compared to OHE baselines' ,fontsize=20)
ax.set_xticks(positions)
ax.set_xticklabels([name for name in dict_names],fontsize=14)

# Adding legend with increased font size
green_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Increase')
red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Decrease')
ax.legend(handles=[green_patch, red_patch], loc='upper left', fontsize='large')

plt.tight_layout()
plt.savefig('results_visualization/violin_plot.png', dpi=300)



