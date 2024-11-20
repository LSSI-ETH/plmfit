import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json

def main():
    # Index labeling
    index_tl_techniques = {0: 'FE', 1: 'LoRA', 2: 'LoRA-', 3: 'Adapters', 4: 'Adapters-'}

    # Colors and markers for each model
    color_marker_dict = {
        'ProteinBERT': ('#FF8C00' ,'*'),
        'ProGen2-small': ('#98FB98', '*'),
        'ProGen2-medium': ('#9ACD32','X'),
        'ProGen2-xlarge': ('#006400', 'D'),
        'ESM2-650M': ('#87CEFA', '*'),
        'ESM2-3B': ('#4169E1', 'X'),
        'ESM2-15B': ('navy', 'D')
    }

    with open("./results/results_matrices.json", "r") as file:
        results_json = json.load(file)
        aav_sampled = results_json["AAV sampled"]
        aav_one_vs_rest = results_json["AAV one-vs-rest"]
        gb1_three_vs_rest = results_json["GB1 three-vs-rest"]
        gb1_one_vs_rest = results_json["GB1 one-vs-rest"]
        meltome_mixed = results_json["Meltome mixed"]

    # Example datasets (replace these with actual data from results_matrices)
    datasets = [
        aav_sampled["best_models"],
        aav_one_vs_rest["best_models"],
        gb1_three_vs_rest["best_models"],
        gb1_one_vs_rest["best_models"],
        meltome_mixed["best_models"],
    ]

    # Baselines for each dataset
    baselines = [
        aav_sampled["ohe_baseline"],
        aav_one_vs_rest["ohe_baseline"],
        gb1_three_vs_rest["ohe_baseline"],
        gb1_one_vs_rest["ohe_baseline"],
        meltome_mixed["ohe_baseline"]
    ]

    # Names for each dataset
    dict_names = [
        'AAV - sampled',
        'AAV - one_vs_rest',
        'GB1 - three_vs_rest',
        'GB1 - one_vs_rest',
        'Meltome - mixed'
    ]

    fig = plt.figure(figsize=(30, 15))
    gs = GridSpec(2, 5, height_ratios=[1, 1])  # 2 rows, 5 columns (adjustable)
    # Add some white space to the left of the whole figure
    # fig.subplots_adjust(left=0.1)

    # Create the subplots for the datasets in the first row
    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
    axes[0].text(
        -0.25, 0.95, f"A", fontsize=40, fontweight="bold", transform=axes[0].transAxes
    )

    # Plot each dataset in the first row
    roman_numerals = ["i", "ii", "iii", "iv", "v"]
    for ax, data, baseline, name, roman in zip(
        axes[:5], datasets, baselines, dict_names, roman_numerals
    ):
        for i in range(len(index_tl_techniques)):
            color = 'lightgray' if i % 2 == 0 else 'white'
            ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.3)

        for model_name, values in data.items():
            x = np.arange(len(values))  # x-axis: position in the list (0-indexed)
            noise = np.random.uniform(-0.3, 0.3, len(values))  # Adding small noise to x positions
            x_noise = x + noise  # Adding noise to x positions
            color, marker = color_marker_dict[model_name]
            ax.scatter(x_noise, values, label=model_name, color=color, marker=marker, s=100)  # Increased marker size

        # Plot the baseline
        ax.axhline(y=baseline, color='red', linestyle='--', label='OHE baseline')
        ax.tick_params(axis='y', labelsize=25)
        ax.set_ylim(0, 1)  # Set the y-axis limit from 0 to 1
        ax.set_xticks(range(len(index_tl_techniques)))
        ax.set_xticklabels([index_tl_techniques[i] for i in range(len(index_tl_techniques))], fontsize=25, rotation=45)
        ax.grid(True)
        ax.set_title(name, fontsize=25)
        ax.set_ylabel('Performance', fontsize=23, labelpad=27)

        ax.text(
            0.9,
            0.03,
            roman,
            fontsize=30,
            fontweight="bold",
            transform=ax.transAxes,
        )

    # Remove y-ticks and labels for all but the first subplot in the first row
    for ax in axes[1:5]:
        ax.set_yticklabels([])
        ax.set_ylabel('')  # Remove y-axis label for these plots

    # Create a single subplot spanning the entire second row
    ax_joint = fig.add_subplot(gs[1, :])

    num_ticks = len(dict_names)
    data_fe = []
    data_ft = []

    # Prepare the data for boxplots
    for dataset, baseline in zip(datasets, baselines):
        # Extract values at the 0th position (for FE)
        values_0 = [values[0] for values in dataset.values()]
        # Compute percentage difference from baseline
        percentage_diff_0 = [(v - baseline) / baseline * 100 for v in values_0]
        data_fe.append(percentage_diff_0)

        # Extract all remaining values (for FT)
        values_remaining = [values[1:] for values in dataset.values()]
        # Flatten the list and compute percentage differences
        percentage_diff_remaining = [(v - baseline) / baseline * 100 for sublist in values_remaining for v in sublist]
        data_ft.append(percentage_diff_remaining)

    # Set positions for the boxplots
    positions_fe = np.arange(num_ticks) - 0.1  # Shift 'FE' boxplot slightly to the left
    positions_ft = np.arange(num_ticks) + 0.1  # Shift 'FT' boxplot slightly to the right

    # Create boxplots with specified colors and closer together
    bp_fe = ax_joint.boxplot(data_fe, positions=positions_fe, widths=0.15, patch_artist=True, 
                            boxprops=dict(facecolor='#0066CC'),
                            medianprops=dict(color='black', linewidth=2))  # Black and wider median line
    bp_ft = ax_joint.boxplot(data_ft, positions=positions_ft, widths=0.15, patch_artist=True, 
                            boxprops=dict(facecolor='#009900'),
                            medianprops=dict(color='black', linewidth=2))  # Black and wider median line

    # Color the area below y=0 in light gray
    ax_joint.axhspan(ax_joint.get_ylim()[0], 0, color='lightgray', alpha=0.5)

    # Evenly distribute x-ticks along the x-axis
    ax_joint.set_xticks(np.arange(num_ticks))  # Set x-ticks for each task
    ax_joint.set_xticklabels(dict_names, fontsize=25, rotation=0)  # Set labels from dict_names

    # Add a grid to the plot (only y-axis)
    ax_joint.tick_params(axis='y', labelsize=23)
    ax_joint.set_ylabel('% difference in performance', fontsize=23)
    ax_joint.grid(axis='y')  # Add only y-axis grid lines

    # Add a red dotted line at y=0
    ax_joint.axhline(y=0, color='red', linestyle='--')
    ax_joint.text(x=num_ticks - 0.5, y=5, s='Baseline', color='black', fontsize=22, horizontalalignment='right')
    ax_joint.text(
        -0.05, 0.95, f"B", fontsize=40, fontweight="bold", transform=ax_joint.transAxes
    )

    # Add a legend for the boxplots
    ax_joint.legend([bp_ft["boxes"][0], bp_fe["boxes"][0]], ['Fine tuning', 'Feature extraction'], loc='lower right', fontsize=25)

    # Add a legend for the subplots in the first row
    handles, labels = axes[0].get_legend_handles_labels()  # Taking handles and labels from one of the axes
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.44), ncol=len(labels), fontsize=22, frameon=False)

    # Adjust the layout for the figure
    plt.subplots_adjust(hspace=0.3)  # Increase space between rows (adjust the 0.3 as needed)
    ax_joint.set_title("                  ", fontsize=55)  # Remove title from the subplot
    plt.tight_layout()

    # Print statistics for each task below the x-axis

    # Save the figure
    plt.savefig('results/til_overview_with_joint_boxplots_and_legend_spaced.png', bbox_inches='tight', dpi=300)

    """ Save the same type of information in a tex file that looks like that:
        \begin{table}
            \caption{\centering Statistical summaries (quarter1, quarter3, median and max) of the box plots depicted in \autoref{fig:3}B.}
            \label{tab:box}
            \centering
            \renewcommand{\arraystretch}{1.5}
            \begin{tabular}{cccc}
            \toprule
            \multicolumn{1}{c}{\multirow{2}{*}{Task}}                  & \multicolumn{1}{c}{\multirow{2}{*}{Box plot stats}} & \multicolumn{1}{c}{\multirow{2}{*}{FE}} & \multirow{2}{*}{FT} \\
            \multicolumn{1}{c}{}                                       & \multicolumn{1}{c}{}                               & \multicolumn{1}{c}{}                    &                     \\ \midrule
            \multicolumn{1}{c}{\multirow{4}{*}{AAV - sampled}}         & Q1                                                  & -5.62\%                                    & -0.39\%                \\
            \multicolumn{1}{c}{}                                       & Q3                                                  & -1.48\%                                     & 7.45\%                 \\
            \multicolumn{1}{c}{}                                       & Median                                              & -3.2\%                                      & 3.95\%                 \\
            \multicolumn{1}{c}{}                                       & Max                                                 & 2.32\%                                      & 7.45\%                 \\ \hline
            \multicolumn{1}{c}{\multirow{4}{*}{AAV - one\_vs\_rest}}   & Q1                                                  & -38.61\%                                    & 8.38\%                 \\
            \multicolumn{1}{c}{}                                       & Q3                                                  & 8.45\%                                      & 47.17\%                \\
            \multicolumn{1}{c}{}                                       & Median                                              & -30.54\%                                    & 38.03\%                \\
            \multicolumn{1}{c}{}                                       & Max                                                 & 8.45\%                                      & 47.17\%                \\ \hline
            \multicolumn{1}{c}{\multirow{4}{*}{GB1 - three\_vs\_rest}} & Q1                                                  & -14.05\%                                    & -8.56\%                \\
            \multicolumn{1}{c}{}                                       & Q3                                                  & -4\%                                       & 4.9\%                  \\
            \multicolumn{1}{c}{}                                       & Median                                              & -5.65\%                                     & 2.89\%                 \\
            \multicolumn{1}{c}{}                                       & Max                                                 & -4\%                                        & 4.9\%                  \\ \hline
            \multicolumn{1}{c}{\multirow{4}{*}{GB1 - one\_vs\_rest}}   & Q1                                                  & -6.41\%                                     & -34.65\%               \\
            \multicolumn{1}{c}{}                                       & Q3                                                  & 37.67\%                                     & 30.98\%                \\
            \multicolumn{1}{c}{}                                       & Median                                              & 31.46\%                                     & 0.74\%                 \\
            \multicolumn{1}{c}{}                                       & Max                                                 & 37.67\%                                     & 30.98\%                \\ \hline
            \multicolumn{1}{c}{\multirow{4}{*}{Meltome - mixed}}       & Q1                                                  & 58.12\%                                     & 31.47\%                \\
            \multicolumn{1}{c}{}                                       & Q3                                                  & 71.27\%                                     & 117.83\%               \\
            \multicolumn{1}{c}{}                                       & Median                                              & 68.71\%                                     & 70.79\%                \\
            \multicolumn{1}{c}{}                                       & Max                                                 & 102.22\%                                    & 117.83\%               \\ \bottomrule
            \end{tabular}
        \end{table}
    """

    with open('results/overleaf/box_plot_stats.tex', 'w') as file:
        file.write("\\begin{table}\n")
        file.write("\t\\caption{\\centering Statistical summaries (Q1, Q3, median, and max) of the box plots depicted in \\autoref{fig:3}B.}\n")
        file.write("\t\\label{tab:box}\n")
        file.write("\t\\centering\n")
        file.write("\t\\renewcommand{\\arraystretch}{1.5}\n")
        file.write("\t\\begin{tabular}{cccc}\n")
        file.write("\t\\toprule\n")
        file.write("\t\\multicolumn{1}{c}{\\multirow{2}{*}{Task}} & \\multicolumn{1}{c}{\\multirow{2}{*}{Stat}} & \\multicolumn{1}{c}{FE} & FT \\\\\n")
        file.write("\t\\multicolumn{1}{c}{} & \\multicolumn{1}{c}{} & \\multicolumn{1}{c}{} & \\\\\n")
        file.write("\t\\midrule\n")

        for task, fe_data, ft_data in zip(dict_names, data_fe, data_ft):
            # Clean task name by escaping underscores
            task_cleaned = task.replace('_', '\\_')

            # Compute statistics for FE and FT
            fe_q1, fe_q3 = np.percentile(fe_data, [25, 75])
            ft_q1, ft_q3 = np.percentile(ft_data, [25, 75])
            fe_median, ft_median = np.median(fe_data), np.median(ft_data)
            fe_max, ft_max = np.max(fe_data), np.max(ft_data)

            # Write statistics for each task
            file.write(f"\t\\multirow{{4}}{{*}}{{{task_cleaned}}} & Q1 & {fe_q1:.2f}\\% & {ft_q1:.2f}\\% \\\\\n")
            file.write(f"\t & Q3 & {fe_q3:.2f}\\% & {ft_q3:.2f}\\% \\\\\n")
            file.write(f"\t & Median & {fe_median:.2f}\\% & {ft_median:.2f}\\% \\\\\n")
            file.write(f"\t & Max & {fe_max:.2f}\\% & {ft_max:.2f}\\% \\\\\n")
            file.write("\t\\hline\n")

        file.write("\t\\end{tabular}\n")
        file.write("\\end{table}\n")


# # Custom colors for legend and bars
# custom_colors = {
#     'Baseline': 'green',
#     'FE': 'gray',
#     'LoRA': 'brown',
#     'LoRA-': 'lightcoral',
#     'Adapters': 'peru',
#     'Adapters-': 'peachpuff'
# }
# # Plotting
# fig, axes = plt.subplots(nrows=1, ncols=len(datasets), figsize=(25, 5), sharey=True)

# for ax, data, baseline, dict_name in zip(axes, datasets, baselines, dict_names):
#     # Extracting data into a DataFrame
#     plot_data = []
#     models = list(data.keys())
#     for model in models:
#         for i, value in enumerate(data[model]):
#             plot_data.append([model, index_tl_techniques[i], value])

#     df = pd.DataFrame(plot_data, columns=['Model', 'Technique', 'Value'])

#     # Prepare data for plotting
#     num_models = len(models)
#     num_techniques = len(index_tl_techniques)
#     bar_width = 0.2  # Width of each bar
#     bar_positions = np.arange(num_models) * 1.5  # Positions for the bars, multiplied by 1.5 for separation

#     # Plotting
#     for i, technique in enumerate(index_tl_techniques.values()):
#         # Calculate x positions for bars in the group
#         x_positions = bar_positions + i * bar_width

#         # Select data for current technique
#         df_tech = df[df['Technique'] == technique]

#         # Plot bars for current technique with custom colors
#         ax.bar(x_positions, df_tech['Value'], width=bar_width, label=technique, alpha=0.7, color=custom_colors[technique])

#     # Plotting the baseline (green line)
#     ax.axhline(y=baseline, color='green', linestyle='--', label='OHE - baseline')

#     # Set x-axis labels for each group of barplots (Protein Language Models)
#     ax.set_xticks(bar_positions + (num_techniques / 2 - 0.5) * bar_width)
#     ax.set_xticklabels(models, rotation=45, ha='right')
#     ax.set_title(name,fontsize = 20)
#     ax.set_xlabel('')

#     #ax.set_title(dict_name.replace('_', ' ').capitalize())

# # Adding common y-axis label
# axes[0].set_ylabel('Performance', fontsize = 16)

# # Combine legends into a single legend outside the subplots
# handles, labels = axes[-1].get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.0), ncol=num_techniques + 1, frameon=False,fontsize='23')

# # Adjust layout
# plt.tight_layout()
# plt.savefig('results_visualization/perf_trainable.png', bbox_inches='tight',dpi=300)

if __name__ == "__main__":
    main()
