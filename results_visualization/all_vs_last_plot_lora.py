import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data
data = {
    "Task": ["AAV-sampled", "GB1-three vs. rest", "Meltome-mixed"],
    "LoRA_Performance": [0.911, 0.876, 0.723],
    "LoRA_GPU_RAM": [28000, 48000, 40000],
    "LoRA_Throughput": [5.8612, 0.2378, 4.1445, 4.1445, 5.9412],
    "LoRA-_Performance": [0.891, 0.858, 0.698],
    "LoRA-_GPU_RAM": [6400, 6500, 12000],
    "LoRA-_Throughput": [1.6009, 2.2957, 0.1115, 0.1506, 3.1855],
}

df = pd.DataFrame(data)

fig, (ax1, ax3) = plt.subplots(nrows=2, ncols=1, figsize=(12, 11), sharex=True)

# First subplot for Performance and Throughput
bar_width = 0.17
index = np.arange(len(df['Task']))

# Plotting performance bars
ax1.bar(index, df['LoRA_Performance'], bar_width, label='LoRA', color='royalblue')
ax1.bar(index + bar_width, df['LoRA-_Performance'], bar_width, label='LoRA-', color='skyblue')

# # Secondary axis for throughput, displayed as black dots
# ax2 = ax1.twinx()
# ax2.plot(index, df['LoRA_Throughput'], 'o', color='black', label='LoRA Throughput')
# ax2.plot(index + bar_width, df['LoRA-_Throughput'], 'o', color='black', label='LoRA- Throughput')

ax1.set_ylabel('Performance')
# ax2.set_ylabel('Throughput (Seconds per Batch)')
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(df['Task'], rotation=45, ha='right')
ax1.legend(loc='upper right')
# ax2.legend(loc='upper right')

# Second subplot for GPU RAM Usage
ax3.bar(index, df['LoRA_GPU_RAM'], bar_width, label='LoRA', color='grey')
ax3.bar(index + bar_width, df['LoRA-_GPU_RAM'], bar_width, label='LoRA-', color='lightgrey')

ax3.set_xlabel('Task')
ax3.set_ylabel('GPU RAM Usage (MiB)')
ax3.legend()
ax3.invert_yaxis()

# Title of figure is: Comparison of memory usage and performance between LoRA and LoRA- on different tasks
fig.suptitle(
    "Comparison of memory usage and performance between LoRA and LoRA- on different tasks",
    fontsize=12,
)


plt.tight_layout()
plt.show()
