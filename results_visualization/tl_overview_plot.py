import results_visualization.results_matrices as results_matrices
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

index_tl_techniqes = {0:'FE', 1: 'LoRA', 2:'LoRA-', 3:'Adapters' , 4:'Adapters-'}

datasets = [
    results_matrices.aav_sampled_dict['best_models'],
    results_matrices.aav_one_vs_rest_dict['best_models'],
    results_matrices.gb1_three_vs_rest_dict['best_models'],
    results_matrices.gb1_one_vs_rest_dict['best_models'],
    results_matrices.meltome_mixed_dict['best_models']
]

# Index labeling
index_tl_techniques = {0: 'FE', 1: 'LoRA', 2: 'LoRA-', 3: 'Adapters', 4: 'Adapters-'}

# Colors and markers for each model
color_marker_dict = {
    'ProteinBERT': ('blue', '*'),
    'ProGen2-small': ('orange', '*'),
    'ProGen2-medium': ('orange', 'X'),
    'ProGen2-xlarge': ('orange', 'D')
}

# Extracting data
best_models = results_matrices.aav_sampled_dict['best_models']

model_names = best_models.keys()
num_points = len(next(iter(best_models.values())))  # Number of points per model (assuming all have the same length)

# Plotting
plt.figure(figsize=(10, 6))

# Plotting the background colors
for i in range(num_points):
    color = 'lightgray' if i % 2 == 0 else 'white'
    plt.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.3)

# Plotting the data with random noise
for model_name, values in best_models.items():
    x = np.arange(num_points)  # x-axis: position in the list (0-indexed)
    noise = np.random.uniform(-0.3, 0.3, num_points)  # Adding small noise to x positions
    x_noise = x + noise  # Adding noise to x positions
    color, marker = color_marker_dict[model_name]
    plt.scatter(x_noise, values, label=model_name, color=color, marker=marker, s=100)  # Increased marker size

# Adding legend
plt.legend(title='Protein Language Models')

# Setting x-axis labels
plt.xticks(ticks=range(num_points), labels=[index_tl_techniques[i] for i in range(num_points)])

# Adding labels and title
plt.xlabel('Techniques')
plt.ylabel('Values')
plt.title('Scatter Plot of Protein Language Models Performance with Noise')
plt.grid(True)
plt.show()
