import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data
data = {
    "Task": ["AAV-sampled", "GB1-three vs. rest", "Meltome-mixed"],
    "Adapters_Performance": [0.917, 0.868, 0.708],
    "Adapters_GPU_RAM": [29000, 52000, 47000],
    "Adapters_Throughput": [4.2948, 0.4427, 6.117],
    "Adapters-_Performance": [0.599, 0.825, 0.279],
    "Adapters-_GPU_RAM": [6500, 7200, 12800],
    "Adapters-_Throughput": [0.0657, 0.817, 0.0456],
}

df = pd.DataFrame(data)

fig, (ax1, ax3) = plt.subplots(nrows=2, ncols=1, figsize=(12, 11), sharex=True)

# First subplot for Performance and Throughput
bar_width = 0.17
index = np.arange(len(df['Task']))

# Plotting performance bars
ax1.bar(index, df['Adapters_Performance'], bar_width, label='Adapters', color='darkgreen')
ax1.bar(index + bar_width, df['Adapters-_Performance'], bar_width, label='Adapters-', color='springgreen')

# # Secondary axis for throughput, displayed as black dots
# ax2 = ax1.twinx()
# ax2.plot(index, df['Adapters_Throughput'], 'o', color='black', label='Adapters Throughput')
# ax2.plot(index + bar_width, df['Adapters-_Throughput'], 'o', color='black', label='Adapters- Throughput')

ax1.set_ylabel('Performance')
# ax2.set_ylabel('Throughput (Seconds per Batch)')
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(df['Task'], rotation=45, ha='right')
ax1.legend(loc='upper right')
# ax2.legend(loc='upper right')

# Second subplot for GPU RAM Usage
ax3.bar(index, df['Adapters_GPU_RAM'], bar_width, label='Adapters', color='grey')
ax3.bar(index + bar_width, df['Adapters-_GPU_RAM'], bar_width, label='Adapters-', color='lightgrey')

ax3.set_xlabel('Task')
ax3.set_ylabel('GPU RAM Usage (MiB)')
ax3.legend()
ax3.invert_yaxis()

# Title of figure is: Comparison of memory usage and performance between adapters and adapters- on different tasks
fig.suptitle('Comparison of memory usage and performance between adapters and adapters- on different tasks', fontsize=12)

plt.tight_layout()
plt.show()
