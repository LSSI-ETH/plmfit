import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, interpolate

def calc_spearman(group, pred_col):
    correlation, p_value = stats.spearmanr(group['score'], group[pred_col])
    return pd.Series({'Correlation': correlation, 'P-value': p_value})

# Load the JSON files
with open('./results_visualization/gb1_one_vs_rest_progen2-small_lora_last_eos_linear_regression_metrics.json', 'r') as file:
    lora_data = json.load(file)

with open('./results_visualization/gb1_one_vs_rest_progen2-medium_bottleneck_adapters_quarter3_mean_linear_regression_metrics.json', 'r') as file:
    ada_data = json.load(file)

with open('./results_visualization/gb1_one_vs_rest_progen2-small_feature_extraction_quarter3_mean_linear_regression_pred_vs_true.json', 'r') as file:
    fe_data = json.load(file)

with open('./results_visualization/gb1_one_vs_rest_linear_regression_pred_vs_true.json', 'r') as file:
    ohe_data = json.load(file)

# Extract the 'pred' array
lora_pred_scores = lora_data['pred_data']['preds']
ada_pred_scores = ada_data['pred_data']['preds']
fe_pred_scores = fe_data['y_pred']
ohe_pred_scores = ohe_data['y_pred']

# Load the dataset
dataset = pd.read_csv('./plmfit/data/gb1/gb1_data_full.csv')

# Filter the dataset for the test split
test_data = dataset[dataset['one_vs_rest'] == 'test']

test_data['lora_pred_score'] = lora_pred_scores
test_data['ada_pred_score'] = ada_pred_scores
test_data['fe_pred_score'] = fe_pred_scores
test_data['ohe_pred_score'] = ohe_pred_scores

# Ensure 'no_mut' is in numerical form and label everything over 20 as 20+
test_data['no_mut'] = test_data['no_mut'].astype(float)

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

def plot_with_error_bar(ax, x, y, p_value, label, color, alpha=0.05):
    print(p_value)
    ax.plot(x, y, label=label, color=color, marker='o')

    # plot p value at minimum at 0.005
    p_value = p_value.apply(lambda x: 0.005 if x < 0.005 else x)
    ax.errorbar(x, y, yerr=p_value, fmt='o', color=color, capsize=2)

plot_with_error_bar(axs[0], correlations_lora.index, correlations_lora['Correlation'], correlations_lora['P-value'], 'LoRA', 'blue')
plot_with_error_bar(axs[0], correlations_ada.index, correlations_ada['Correlation'], correlations_ada['P-value'], 'Ada', 'orange')
plot_with_error_bar(axs[0], correlations_fe.index, correlations_fe['Correlation'], correlations_fe['P-value'], 'FE', 'green')
plot_with_error_bar(axs[0], correlations_ohe.index, correlations_ohe['Correlation'], correlations_ohe['P-value'], 'OHE', 'red')

axs[0].set_xlabel('Edit Distance')
axs[0].set_ylabel("Spearman's Correlation")
axs[0].set_title("Spearman's Correlation by Edit Distance")
str_month_list = ['2','3','4']
axs[1].set_xticks([2, 3, 4])
axs[1].set_xticklabels(str_month_list)
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
str_month_list = ['2','3','4']
axs[1].set_xticks([2, 3, 4])
axs[1].set_xticklabels(str_month_list)

plt.tight_layout()
plt.show()