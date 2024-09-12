import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Data dictionaries
total = {
    "accuracy": 0.9169,
    "mcc": 0.8111,
    "recall": 0.9272,
    "precision": 0.95,
    "f1": 0.9385
}

monotone = {
    "accuracy": 0.9422,
    "mcc": 0.8651,
    "recall": 0.9421,
    "precision": 0.9754,
    "f1": 0.9585,
    "micro_hit_rate": 0.9417,
    "full_predicted_rate": 0.8905
}

varied = {
    "accuracy": 0.7327,
    "mcc": 0.4658,
    "recall": 0.7775,
    "precision": 0.7213,
    "f1": 0.7484,
    "micro_hit_rate": 0.7252,
    "full_predicted_rate": 0.4618
}

# Creating data dictionary and ensuring common keys only
common_keys = set(total.keys()).intersection(set(monotone.keys())).intersection(set(varied.keys()))

# Filtering dictionaries to include only common keys
total_filtered = {k: total[k] for k in common_keys}
monotone_filtered = {k: monotone[k] for k in common_keys}
varied_filtered = {k: varied[k] for k in common_keys}

# Creating a DataFrame for seaborn plotting
data = []
for key in common_keys:
    data.append({"Metric": key, "Value": total_filtered[key], "Type": "Total"})
    data.append({"Metric": key, "Value": monotone_filtered[key], "Type": "Monotone"})
    data.append({"Metric": key, "Value": varied_filtered[key], "Type": "Varied"})

df = pd.DataFrame(data)

# Plotting using seaborn
plt.figure(figsize=(12, 6))
bar_plot = sns.barplot(data=df, x='Metric', y='Value', hue='Type', palette=['#808080', '#ff7f0e', '#006400'])

# Setting the width of the bars
for patch in bar_plot.patches:
    current_width = patch.get_width()
    diff = current_width - 0.25
    # we change the bar width
    patch.set_width(0.25)
    # we recenter the bar
    patch.set_x(patch.get_x() + diff * .5)

plt.title('"Varied" vs "Monotone" pefromance metrics')

plt.ylabel('Metrics value', fontsize = 14)
plt.xlabel('')
plt.legend( loc='upper right', ncol=3)

plt.show()
import matplotlib.pyplot as plt

# Example data
models = ['Custom (pwff)','Custom (no pwff)' , 'Ankh-based', 'OHE - Baseline', 'MLP']
metric1 = [0.89, 0.85,  0.92, 0.81, 0.82]  # e.g., accuracy
metric2 = [0.47,0.44, 0.4, 0.36, 0.39]  # e.g., precision

plt.figure(figsize=(10, 6))
colors = ['darkblue', 'blue','green', 'red', 'orange']
markers = ['*','o', '^', 's' , 's']
sizes = [150,80,80, 80, 80]  # Adjust sizes as needed

for i, (m1, m2, model) in enumerate(zip(metric1, metric2, models)):
    plt.scatter(m1, m2, color=colors[i], marker=markers[i], s=sizes[i], label=model)

for i, model in enumerate(models):
    plt.text(metric1[i] - 0.02, metric2[i], model, fontsize=12, ha='right')

plt.xlabel('Monotone')
plt.ylabel('Varied')
plt.title('FPR for "varied" and "monotone" groups')

# Draw diagonal line
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

plt.grid(True)
plt.xlim(0.3, 1.0)  # Adjust x-axis limits as needed
plt.ylim(0.3, 0.6)  # Adjust y-axis limits as needed

plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data (replace this with your actual data)
data = {
    'Model': ['OHE Baseline', 'MLP', 'CNN', 'ESM-based', 'ESM-Fine-tuned', 'Ankh-based', 
              'Custom Transformer (w/o PWFF)', 'Custom Transformer (with PWFF)'],
    'MCC': [0.73, 0.77, 0.78, 0.81, 0.79, 0.83, 0.8, 0.82],
    'FPR': [0.78, 0.79, 0.84, 0.85, 0.84, 0.86, 0.83, 0.86],
}

df = pd.DataFrame(data)

# Melt the dataframe to 'long' format for easier plotting
df_long = pd.melt(df, id_vars=['Model'], var_name='Metric', value_name='Performance')

# Define colors for each metric (adjustable)
metric_colors = {
    'MCC': '#808080',  # Gray color for MCC
    'FPR': '#FFA500',  # Orange color for FPR
}

# Adjustable parameters
bar_width = 0.35  # Width of the bars
bar_spacing = 0  # Increased spacing between groups of bars
sns_palette = [metric_colors[key] for key in df_long['Metric'].unique()]  # Custom color palette

# Plot using Seaborn (flipped orientation)
plt.figure(figsize=(10, 8))
ax = sns.barplot(x='Performance', y='Model', hue='Metric', data=df_long,
                 palette=sns_palette, edgecolor=None, linewidth=0)  # Set edgecolor to None and linewidth to 0

# Print values on bars
for p in ax.patches:
    ax.annotate(f'{p.get_width():.2f}', ((p.get_width() - 0.02), p.get_y() - 0.05+ p.get_height()/2), color='white', weight='bold')

# Adjusting legend and labels
plt.legend(loc='lower right', fontsize='large')  # Increase the legend font size
plt.title('Performance evaluation')
plt.xlabel('')  # Empty x-axis label
plt.ylabel('')  # Empty y-axis label

# Hide x-axis
plt.gca().axes.get_xaxis().set_visible(False)

# Adjust y-axis position and width of bars
num_models = len(df['Model'])
plt.yticks(ticks=range(num_models), labels=df['Model'])
bar_positions = [i + j * (bar_width + bar_spacing) for i in range(num_models) for j in range(len(metric_colors))]
plt.ylim(-0.5, num_models - 0.5 + (len(metric_colors) - 1) * (bar_width + bar_spacing))

# Show plot
plt.xlim(0.6)
plt.tight_layout()
plt.show()
