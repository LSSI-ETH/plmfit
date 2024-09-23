import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, interpolate

def calc_spearman(group, pred_col):
    correlation, p_value = stats.spearmanr(group['score'], group[pred_col])
    return pd.Series({'Correlation': correlation, 'P-value': p_value})


file_name_lora = 'aav_one_vs_many_progen2-small_lora_last_mean_linear_regression_metrics.json'
file_name_ada = 'aav_one_vs_many_progen2-small_bottleneck_adapters_last_mean_linear_regression_metrics.json'
file_name_fe = 'aav_one_vs_many_esm2_t48_15B_UR50D_feature_extraction_quarter1_mean_linear_regression_pred_vs_true.json'
file_name_ohe = 'aav_one_vs_many_mlp_regression_pred_vs_true.json'

# Load the JSON files
with open(f'./results_visualization/{file_name_lora}', 'r') as file:
    lora_data = json.load(file)

with open(f'./results_visualization/{file_name_ada}', 'r') as file:
    ada_data = json.load(file)

with open(f'./results_visualization/{file_name_fe}', 'r') as file:
    fe_data = json.load(file)

with open(f'./results_visualization/{file_name_ohe}', 'r') as file:
    ohe_data = json.load(file)

# Create a DataFrame with the file names
file_names_df = pd.DataFrame([{
    'LoRA': file_name_lora,
    'Ada': file_name_ada,
    'FE': file_name_fe,
    'OHE': file_name_ohe
}])

# Extract the 'pred' array
lora_pred_scores = lora_data['pred_data']['preds']
ada_pred_scores = ada_data['pred_data']['preds']
fe_pred_scores = fe_data['y_pred']
ohe_pred_scores = ohe_data['y_pred']

# Load the dataset
dataset = pd.read_csv('./plmfit/data/aav/aav_data_full.csv')

# Filter the dataset for the test split
test_data = dataset[dataset['one_vs_many'] == 'test']

test_data['lora_pred_score'] = lora_pred_scores
test_data['ada_pred_score'] = ada_pred_scores
test_data['fe_pred_score'] = fe_pred_scores
test_data['ohe_pred_score'] = ohe_pred_scores

# Ensure 'no_mut' is in numerical form and label everything over 20 as 20+
test_data['no_mut'] = test_data['no_mut'].astype(float)
test_data['no_mut'] = test_data['no_mut'].apply(lambda x: 20 if x > 20 else x)

cols_to_keep = ['no_mut', 'score', 'lora_pred_score', 'ada_pred_score', 'fe_pred_score', 'ohe_pred_score']
test_data = test_data[cols_to_keep]

# Calculate Spearman correlations for each edit distance group
correlations_lora = test_data.groupby('no_mut').apply(calc_spearman, 'lora_pred_score')
correlations_ada = test_data.groupby('no_mut').apply(calc_spearman, 'ada_pred_score')
correlations_fe = test_data.groupby('no_mut').apply(calc_spearman, 'fe_pred_score')
correlations_ohe = test_data.groupby('no_mut').apply(calc_spearman, 'ohe_pred_score')

# Interpolate for smoothing
def interpolate_smooth(x, y, kind='cubic', num=500):
    x_smooth = np.linspace(x.min(), x.max(), num)
    y_smooth = interpolate.interp1d(x, y, kind=kind)(x_smooth)
    return x_smooth, y_smooth

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

def plot_with_fill(ax, x, y, p_value, label, color, alpha=0.05):
    x_smooth, y_smooth = interpolate_smooth(x, y)
    _, y_lower = interpolate_smooth(x, y - p_value*0.3)
    _, y_upper = interpolate_smooth(x, y + p_value*0.3)
    ax.plot(x, y, label=label, color=color, marker='o')
    ax.fill_between(x_smooth, y_lower, y_upper, color=color, alpha=alpha)

plot_with_fill(axs[0], correlations_lora.index, correlations_lora['Correlation'], correlations_lora['P-value'], 'LoRA', 'blue')
plot_with_fill(axs[0], correlations_ada.index, correlations_ada['Correlation'], correlations_ada['P-value'], 'Ada', 'orange')
plot_with_fill(axs[0], correlations_fe.index, correlations_fe['Correlation'], correlations_fe['P-value'], 'FE', 'green')
plot_with_fill(axs[0], correlations_ohe.index, correlations_ohe['Correlation'], correlations_ohe['P-value'], 'OHE', 'red')

axs[0].set_xlabel('Edit Distance')
axs[0].set_ylabel("Spearman's Correlation")
axs[0].set_title("Spearman's Correlation by Edit Distance")
str_month_list = ['2','5','10','15','20+']
axs[0].set_xticks([2, 5, 10, 15, 20])
axs[0].set_xticklabels(str_month_list)
axs[0].legend()

# Subplot for counts
counts = test_data['no_mut'].value_counts().sort_index()
bars = axs[1].bar(counts.index, counts.values, label='Count')

# Add counts above the bars
for bar in bars:
    height = bar.get_height()
    axs[1].text(bar.get_x() + bar.get_width() / 2, height, '%d' % int(height), ha='center', va='bottom', fontsize=8)

axs[1].set_xlabel('Edit Distance')
axs[1].set_ylabel('Count')
axs[1].set_title('Number of Entries by Edit Distance')
axs[1].legend()
str_month_list = ['2','5','10','15','20+']
axs[1].set_xticks([2, 5, 10, 15, 20])
axs[1].set_xticklabels(str_month_list)

correlations_lora.rename(
    columns={'Correlation': 'LoRA', 'P-value': 'LoRA-P'}, inplace=True)
correlations_ada.rename(
    columns={'Correlation': 'Ada', 'P-value': 'Ada-P'}, inplace=True)
correlations_fe.rename(
    columns={'Correlation': 'FE', 'P-value': 'FE-P'}, inplace=True)
correlations_ohe.rename(
    columns={'Correlation': 'OHE', 'P-value': 'OHE-P'}, inplace=True)

# Combine all Spearmans into a single DataFrame
combined_corr = pd.concat(
    [correlations_lora, correlations_ada, correlations_fe, correlations_ohe, counts], axis=1)

combined_corr_with_filenames = pd.concat(
    [file_names_df, combined_corr], ignore_index=False)

# Save the combined DataFrame to a CSV file
combined_corr_with_filenames.to_csv('./results/aav_corr_by_edit_distance.csv')

print(combined_corr)
print(counts)

plt.tight_layout()
plt.show()
