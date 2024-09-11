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
    'Adapters_Performance': [0.879, 0.416, 0.928, 0.827, 0.709, 0.370, 0.477],
    'Adapters_GPU_RAM': [4659, 5329, 30765, 6197, 39339, 7797, 1808],
    # 'Adapters_Throughput': [0.4427, 0.6432, 4.2948, 0.2817, 6.117],
    'Adapters-_Performance': [0.862, 0.382, 0.909, 0.757, 0.693, 0.308, 0.549],
    'Adapters-_GPU_RAM': [3863, 15282, 1565, 9473, 29929, 1667, 1263],
    # 'Adapters-_Throughput': [0.817, 0.1566, 0.0657, 1.6368, 0.0456]
}

df = pd.DataFrame(data)

fig, (ax1, ax3) = plt.subplots(nrows=2, ncols=1, figsize=(14, 16), sharex=True)

# First subplot for Performance and Throughput
bar_width = 0.35
index = np.arange(len(df['Task']))

# Plotting performance bars
ax1.bar(index, df['Adapters_Performance'], bar_width, label='Adapters Performance', color='coral')
ax1.bar(index + bar_width, df['Adapters-_Performance'], bar_width, label='Adapters- Performance', color='mistyrose')

# Secondary axis for throughput, displayed as black dots
# ax2 = ax1.twinx()
# ax2.plot(index, df['Adapters_Throughput'], 'o', color='black', label='Adapters Throughput')
# ax2.plot(index + bar_width, df['Adapters-_Throughput'], 'o', color='black', label='Adapters- Throughput')

ax1.set_ylabel('Performance')
# ax2.set_ylabel('Throughput (Seconds per Batch)')
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(df['Task'], rotation=45, ha='right')
ax1.legend(loc='upper left')
# ax2.legend(loc='upper right')

# Second subplot for GPU RAM Usage
ax3.bar(index, df['Adapters_GPU_RAM'], bar_width, label='Adapters GPU RAM', color='grey')
ax3.bar(index + bar_width, df['Adapters-_GPU_RAM'], bar_width, label='Adapters- GPU RAM', color='lightgrey')

ax3.set_xlabel('Task')
ax3.set_ylabel('GPU RAM Usage (MB)')
ax3.legend()
ax3.invert_yaxis()

plt.tight_layout()
plt.show()
