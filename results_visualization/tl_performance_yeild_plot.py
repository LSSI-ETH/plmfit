import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json

def main():
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

    # Prepare the data for plotting
    plot_data = []
    for data_dict, baseline, dict_name in zip(datasets, baselines, dict_names):
        for model, values in data_dict.items():
            for value in values:
                percentage_diff = (value - baseline) / baseline * 100
                plot_data.append([dict_name, model, percentage_diff])

    df = pd.DataFrame(plot_data, columns=['Task', 'Model', 'PercentageDiff'])

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 6))

    # Create violin plots for each task
    positions = np.arange(len(dict_names))
    legend_labels = []

    for i, task in enumerate(dict_names):
        task_data = df[df['Task'] == task]['PercentageDiff']

        # Splitting data for positive and negative regions
        positive_data = task_data[task_data >= 0]
        negative_data = task_data[task_data < 0]

        # Check if data is not empty before plotting
        if not positive_data.empty:
            parts_positive = ax.violinplot(positive_data, positions=[i], showmeans=False, showmedians=True)
            for pc in parts_positive['bodies']:
                pc.set_facecolor('green')
                pc.set_edgecolor('black')
                pc.set_linewidth(1)  # Set edge linewidth to 1
                pc.set_alpha(0.7)

            # Annotate maximum value above violin plot
            max_value = np.max(positive_data)
            ax.text(i, max_value + 0.5, f'{max_value:.2f}%', ha='center', va='bottom', color='green', fontsize=10)

            legend_labels.append('Increase')

        if not negative_data.empty:
            parts_negative = ax.violinplot(negative_data, positions=[i], showmeans=False, showmedians=True)
            for pc in parts_negative['bodies']:
                pc.set_facecolor('red')
                pc.set_edgecolor('black')
                pc.set_linewidth(1)  # Set edge linewidth to 1
                pc.set_alpha(0.7)

            # Annotate minimum value below violin plot
            min_value = np.min(negative_data)
            ax.text(i, min_value - 0.9, f'{min_value:.2f}%', ha='center', va='top', color='red', fontsize=10)

            legend_labels.append('Decrease')

        # Customizing lines (mean, median, max, min)
        for partname in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
            if not positive_data.empty:
                vp = parts_positive[partname]
                vp.set_edgecolor('black')  # Set edge color to black
                vp.set_linewidth(0.5)  # Set linewidth to 0.5
            if not negative_data.empty:
                vn = parts_negative[partname]
                vn.set_edgecolor('black')  # Set edge color to black
                vn.set_linewidth(0.5)  # Set linewidth to 0.5

    # Set labels and title
    # ax.set_xlabel('Tasks')
    ax.set_ylabel('% difference in performance',fontsize=15)
    ax.set_title(' Per task TL performance yield compared to OHE baselines' ,fontsize=20)
    ax.set_xticks(positions)
    ax.set_xticklabels([name for name in dict_names],fontsize=14)

    # Adding legend with increased font size
    green_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Increase')
    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Decrease')
    ax.legend(handles=[green_patch, red_patch], loc='upper left', fontsize='large')

    plt.tight_layout()
    plt.savefig('results/violin_plot.png', dpi=300)

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

    with open("results/overleaf/violin_plot_stats.tex", "w") as file:
        file.write("\\begin{table}\n")
        file.write(
            "\t\\caption{\\centering Statistical summaries (Q1, Q3, median, and max) of the violin plots depicted in \\autoref{fig:3}B.}\n"
        )
        file.write("\t\\label{tab:violin}\n")
        file.write("\t\\centering\n")
        file.write("\t\\renewcommand{\\arraystretch}{1.5}\n")
        file.write("\t\\begin{tabular}{cccc}\n")
        file.write("\t\\toprule\n")
        file.write("\tTask & Stat & Increase & Decrease \\\\\n")
        file.write("\t\\midrule\n")

        for task in dict_names:
            # Replace underscores with \_ for proper LaTeX rendering
            task_cleaned = task.replace("_", "\\_")

            task_data = df[df["Task"] == task]["PercentageDiff"]
            q1 = np.percentile(task_data, 25)
            q3 = np.percentile(task_data, 75)
            median = np.median(task_data)
            max_value = np.max(task_data)
            min_value = np.min(task_data)

            # Write task row using multirow
            file.write(
                f"\t\\multirow{{4}}{{*}}{{{task_cleaned}}} & Q1 & {q1:.2f}\\% & {min_value:.2f}\\% \\\\\n"
            )
            file.write(f"\t & Q3 & {q3:.2f}\\% & {max_value:.2f}\\% \\\\\n")
            file.write(f"\t & Median & {median:.2f}\\% & {median:.2f}\\% \\\\\n")
            file.write(f"\t & Max & {max_value:.2f}\\% & {min_value:.2f}\\% \\\\\n")
            file.write("\t\\hline\n")

        file.write("\t\\end{tabular}\n")
        file.write("\\end{table}\n")


if __name__ == "__main__":
    main()
