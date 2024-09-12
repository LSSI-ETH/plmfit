import matplotlib.pyplot as plt
import numpy as np

# Define tasks, setups, and model names
tasks = ['AAV - sampled', 'GB1 - three_vs_rest', 'Meltome - mixed']
setups = ['LoRA', 'LoRA-']
model_names = ['progen2-xlarge', 'esm2_t33_650M_UR50D', 'esm2_t48_15B_UR50D', 'esm2_t48_15B_UR50D', 'progen2-xlarge', 'progen2-xlarge']

# Provided performance and memory values
performance_values = [0.925, 0.904, 0.877, 0.858, 0.723, 0.698]
memory_requirements = [-39237, -2004, -26169, -14771, -39121, -11845]

# Scale memory requirements
scaled_memory_requirements = np.array(memory_requirements) / max(abs(np.array(memory_requirements))) * max(performance_values)

# Define positions of tasks on x-axis
x = np.array([0, 0.25, 0.5])

# Bar width (reduced for closer grouping of tasks)
bar_width = 0.05 # Reduced bar width

# Spacing between the groups of bars (closer grouping)
group_spacing = 0.5

# Colors for the bars
color_dark_green = '#009900'
color_dark_yellow = '#FFCD00'
color_dark_gray = '#4f4f4f'

# Create the plot
fig, ax = plt.subplots(figsize=(25,25))

# Plot bars for performance (above the line)
for i in range(len(setups)):
    # Alternating between green and yellow for performance values
    color_above = color_dark_green
    ax.bar(x + i * bar_width, performance_values[i::len(setups)], bar_width,
           color=color_above, alpha=1 if i % 2 == 0 else 0.3, label=f'Performance {setups[i]}')

# Plot bars for memory (below the line) with alternating alpha
for i in range(len(setups)):
    ax.bar(x + i * bar_width, scaled_memory_requirements[i::len(setups)], bar_width,
           color=color_dark_gray, alpha=1 if i % 2 == 0 else 0.3)

# Customize the x-axis
ax.set_xticks(x )  # Reduced spacing
ax.set_xticklabels(tasks, fontsize=35)  # Increased font size for xticks

ax.set_yticks([1.00 , 0.75 , 0.5 , 0.25, -0.25, -0.5,-0.75,-0.925])  # Reduced spacing
ax.set_yticklabels([1.00 , 0.75 , 0.5 , 0.25, 10000, 20000, 30000,36500], fontsize=25)  # Increased font size for xticks
#ax.set_yticks([1.00 , 0.75 , 0.5 , 0.25, -0.0472, -0.279,-0.348,-0.616,-0.922, -0.925])  # Reduced spacing
#ax.set_yticklabels([1.00 , 0.75 , 0.5 , 0.25, 2004, 11845,14771,26169, 39121,39237], fontsize=25) 
# Add a horizontal line at y=0
ax.axhline(0, color='black', linewidth=2)

# Labels and title with larger font size
ax.tick_params(axis='y', labelsize=25)  # Increas
ax.set_ylabel('RAM memory requirements (MB)                                             Performance  ', fontsize=35)
ax.set_title('Comparison of memory usage for LoRA and LoRA-', fontsize=40)

# Legend for performance bars
legend_labels = ['LoRA', 'LoRA- ']
handles = [plt.Rectangle((0, 0), 1, 1, color=color_dark_green, alpha=1),
           plt.Rectangle((0, 0), 1, 1, color=color_dark_green, alpha=0.3),
           plt.Rectangle((0, 0), 1, 1, color=color_dark_yellow, alpha=1),
           plt.Rectangle((0, 0), 1, 1, color=color_dark_yellow, alpha=0.3)]

ax.legend(handles, legend_labels, loc='upper right', fontsize=35)  # Increased font size for legend

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
