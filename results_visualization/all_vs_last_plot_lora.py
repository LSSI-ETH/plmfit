import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data
data = {
    'Task': [
        'GB1 (three vs rest)', 'GB1 (one vs rest)',
        'AAV (sampled)', 'AAV (one vs rest)',
        'Meltome (mixed)', 'HerH3 (one vs rest)', 'RBD (one vs rest)'
    ],
    'LoRA_Performance': [0.877, 0.405, 0.926, 0.831, 0.723, 0.387, 0.554],
    'LoRA_GPU_RAM': [26169, 12641, 39237, 29631, 39121, 30789, 1645],
    # 'LoRA_Throughput': [5.8612, 0.2378, 4.1445, 4.1445, 5.9412],
    'LoRA-_Performance': [0.858, 0.435, 0.905, 0.770, 0.699, 0.390, 0.536],
    'LoRA-_GPU_RAM': [14771, 6454, 2004, 2447, 11845, 4440, 9290],
    # 'LoRA-_Throughput': [1.6009, 2.2957, 0.1115, 0.1506, 3.1855]
}

df = pd.DataFrame(data)

fig, (ax1, ax3) = plt.subplots(nrows=2, ncols=1, figsize=(14, 16), sharex=True)

# First subplot for Performance and Throughput
bar_width = 0.35
index = np.arange(len(df['Task']))

# Plotting performance bars
ax1.bar(index, df['LoRA_Performance'], bar_width, label='LoRA Performance', color='royalblue')
ax1.bar(index + bar_width, df['LoRA-_Performance'], bar_width, label='LoRA- Performance', color='skyblue')

# # Secondary axis for throughput, displayed as black dots
# ax2 = ax1.twinx()
# ax2.plot(index, df['LoRA_Throughput'], 'o', color='black', label='LoRA Throughput')
# ax2.plot(index + bar_width, df['LoRA-_Throughput'], 'o', color='black', label='LoRA- Throughput')

ax1.set_ylabel('Performance')
# ax2.set_ylabel('Throughput (Seconds per Batch)')
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(df['Task'], rotation=45, ha='right')
ax1.legend(loc='upper left')
# ax2.legend(loc='upper right')

# Second subplot for GPU RAM Usage
ax3.bar(index, df['LoRA_GPU_RAM'], bar_width, label='LoRA GPU RAM', color='grey')
ax3.bar(index + bar_width, df['LoRA-_GPU_RAM'], bar_width, label='LoRA- GPU RAM', color='lightgrey')

ax3.set_xlabel('Task')
ax3.set_ylabel('GPU RAM Usage (MB)')
ax3.legend()
ax3.invert_yaxis()

plt.tight_layout()
plt.show()
