import os
import glob
import pandas as pd
import json

# Mapping dictionary for task names
task_mapping = {
    "aav_one_vs_many": "AAV one-vs-rest",
    "aav_sampled": "AAV sampled",
    "gb1_one_vs_rest": "GB1 one-vs-rest",
    "gb1_three_vs_rest": "GB1 three-vs-rest",
    "meltome_mixed": "Meltome mixed",
    "herH3_one_vs_rest": "HER2 one-vs-rest",
    "rbd_one_vs_rest": "RBD one-vs-rest",
    "ss3_sampled": "SS3 sampled",
}

task_reverse_mapping = {v: k for k, v in task_mapping.items()}

# List of fine-tuning techniques and their abbreviations
fine_tuning_techniques = {
    "Feature Extraction_linear": "FE-linear",
    "Feature Extraction_mlp": "FE-mlp",
    "LoRA (All Layers)": "LoRA",
    "LoRA- (Last Layer)": "LoRA-",
    "Adapters (All Layers)": "Adapters",
    "Adapters- (Last Layer)": "adapters-",
}

plm_mapping = {
    "proteinbert": "ProteinBERT",
    "progen2-small": "ProGen2-small",
    "progen2-medium": "ProGen2-medium",
    "progen2-xlarge": "ProGen2-xlarge",
    "esm2_t33_650M_UR50D": "ESM2-650M",
    "esm2_t36_3B_UR50D": "ESM2-3B",
    "esm2_t48_15B_UR50D": "ESM2-15B",
}

fine_tuning_techniques_reverse = {v: k for k, v in fine_tuning_techniques.items()}

# Desired order for PLMs and TLs
plm_order = [
    "ProteinBERT",
    "ProGen2-small",
    "ProGen2-medium",
    "ESM2-650M",
    "ESM2-3B",
    "ProGen2-xlarge",
    "ESM2-15B",
]
tl_order = ["FE-linear", "FE-mlp", "LoRA", "LoRA-", "Adapters", "adapters-"]
layer_order = ["first", "quarter1", "middle", "quarter3", "last"]


def parse_csv_files(directory):
    # Find all CSV files in the specified directory
    csv_files = glob.glob(os.path.join(directory, "*.csv"))

    parsed_results = []

    for csv_file in csv_files:
        # Read the CSV file using pandas
        df = pd.read_csv(csv_file)

        # Check if the CSV file has the expected columns
        expected_columns = [
            "File",
            "Model Name",
            "Layer",
            "Head Type",
            "Model + Head",
        ]
        if all(column in df.columns for column in expected_columns) and (
            "Spearman's Rank Correlation" in df.columns or "MCC" in df.columns
        ):
            # Parse the task name from the 'File' column
            def get_task(file_name):
                for key in task_mapping:
                    if key in file_name:
                        return task_mapping[key]
                return "Unknown"

            # Parse the fine-tuning technique from the file name
            def get_fine_tuning_technique(file_name):
                for technique in fine_tuning_techniques:
                    if technique in file_name:
                        return fine_tuning_techniques[technique]
                return "Unknown"

            def get_plm_name(model_name):
                for model in plm_mapping:
                    if model in model_name:
                        return plm_mapping[model]
                return "Unknown"

            if get_task(os.path.basename(csv_file)) == "SS3 sampled":
                df.rename(
                    columns={"Accuracy": "Metric"},
                    inplace=True,
                )
            else:
                df.rename(
                    columns={"Spearman's Rank Correlation": "Metric", "MCC": "Metric"},
                    inplace=True,
                )
            df["Task"] = df["File"].apply(get_task)
            df["TL"] = get_fine_tuning_technique(os.path.basename(csv_file))
            df["Model Name"] = df["Model Name"].apply(get_plm_name)
            parsed_results.append(df)
        else:
            print(f"Skipping file {csv_file} as it does not have the expected columns")

    return parsed_results


def generate_summary_table(parsed_results):
    summary_tables = {}

    for task in task_mapping.values():
        task_results = [df[df["Task"] == task] for df in parsed_results]
        task_results = pd.concat(task_results)
        summary_table = (
            task_results.groupby(["Model Name", "TL"])["Metric"]
            .max()
            .unstack()
            .fillna(0)
        )

        # Reorder the PLMs and TLs
        summary_table = summary_table.reindex(index=plm_order, columns=tl_order)
        summary_tables[task] = summary_table

    return summary_tables


def generate_layer_analysis(parsed_results):
    layer_analysis = {}

    for task in task_mapping.values():
        task_results = [df[df["Task"] == task] for df in parsed_results]
        task_results = pd.concat(task_results)

        for tl in tl_order:
            tl_results = task_results[task_results["TL"] == tl]
            layer_table = (
                tl_results.groupby(["Model Name", "Layer"])["Metric"]
                .max()
                .unstack()
                .fillna(0)
            )

            # Reorder the PLMs and layers
            layer_table = layer_table.reindex(index=plm_order, columns=layer_order)
            if task not in layer_analysis:
                layer_analysis[task] = {}
            layer_analysis[task][tl] = layer_table

    return layer_analysis


## Define the function to find the best model and layer combo for each task and for each TL
def make_scoreboard(parsed_results):
    # Load CSV 'combined_results.csv' which has GPU and CPU information for each experiment
    combined_results = pd.read_csv("./results_visualization/combined_results.csv")

    results = []

    for task in task_mapping.values():
        task_results = [df[df["Task"] == task] for df in parsed_results]
        task_results = pd.concat(task_results)

        for tl in tl_order:
            tl_results = task_results[task_results["TL"] == tl]
            max_metric = tl_results["Metric"].max()
            if len(tl_results) == 0:
                continue
            best_model = tl_results[tl_results["Metric"] == max_metric][
                "Model Name"
            ].values[0]
            best_layer = tl_results[tl_results["Metric"] == max_metric]["Layer"].values[
                0
            ]
            best_pooling = "-"
            if task != "SS3 sampled":
                best_pooling = tl_results[tl_results["Metric"] == max_metric][
                    "Reduction"
                ].values[0]
            best_head = tl_results[tl_results["Metric"] == max_metric][
                "Head Type"
            ].values[0]
            if task == "SS3 sampled":
                metric_name = "Macro Accuracy"
            else: 
                metric_name = (
                    "Spearman's Corr."
                    if "regression"
                    in tl_results[tl_results["Metric"] == max_metric]["File"].values[0]
                    else "MCC"
                )

            # Find the max GPU usage for the best model and layer combo
            gpu_usage_row = combined_results[
                (
                    combined_results["great_grandparent_folder"]
                    == fine_tuning_techniques_reverse[tl]
                )
                & (
                    combined_results["grandparent_folder"].str.contains(
                        f"{task_reverse_mapping[task]}_{best_model}_bottleneck_adapters_{best_layer}"
                    )
                    | combined_results["grandparent_folder"].str.contains(
                        f"{task_reverse_mapping[task]}_{best_model}_lora_{best_layer}"
                    )
                    | combined_results["grandparent_folder"].str.contains(
                        f"{task_reverse_mapping[task]}_{best_model}_feature_extraction_{best_layer}"
                    )
                )
            ]

            gpu_usage = (
                gpu_usage_row["gpu_max_usage"].values[0]
                if not gpu_usage_row.empty
                else "N/A"
            )

            results.append(
                {
                    "Task": task,
                    "Score": max_metric,
                    "Metric": metric_name,
                    "PLM": best_model,
                    "TL-method": tl,
                    "Layers used": best_layer,
                    "Pooling": best_pooling,  # Assuming pooling is 'Mean', adjust as necessary
                    "Downstream head": best_head,  # Assuming downstream head is 'Linear', adjust as necessary
                    "GPU usage": gpu_usage,
                }
            )

    # Based on 'Task', keep the best performing combo only
    best_results_per_task = []
    for task in task_mapping.values():
        task_results = [result for result in results if result["Task"] == task]
        best_result = max(task_results, key=lambda x: x["Score"])
        best_results_per_task.append(best_result)

    with open("./results/results_matrices.json", "r") as file:
        results_json = json.load(file)

    for task in task_mapping.values():
        task_results = [df[df["Task"] == task] for df in parsed_results]
        task_results = pd.concat(task_results)
        for plm in plm_order:
            results_json[task]["feature_extraction"][plm] = []
            results_json[task]["adapters"][plm] = []
            results_json[task]["lora"][plm] = []
            results_json[task]["best_models"][plm] = []
            for layer in layer_order:
                fe_max = task_results[
                    (task_results["Task"] == task)
                    & (task_results["Model Name"] == plm)
                    & (
                        (task_results["TL"] == "FE-linear")
                        | (task_results["TL"] == "FE-mlp")
                    )
                    & (task_results["Layer"] == layer)
                ]["Metric"].max()
                results_json[task]["feature_extraction"][plm].append(
                    fe_max if not pd.isna(fe_max) else 0
                )
                adapter_max = task_results[
                    (task_results["Task"] == task)
                    & (task_results["Model Name"] == plm)
                    & (
                        (task_results["TL"] == "Adapters")
                        | (task_results["TL"] == "adapters-")
                    )
                    & (task_results["Layer"] == layer)
                ]["Metric"].max()
                results_json[task]["adapters"][plm].append(
                    adapter_max if not pd.isna(adapter_max) else 0
                )
                lora_max = task_results[
                    (task_results["Task"] == task)
                    & (task_results["Model Name"] == plm)
                    & (task_results["TL"] == "LoRA")
                    & (task_results["Layer"] == layer)
                ]["Metric"].max()
                results_json[task]["lora"][plm].append(
                    lora_max if not pd.isna(lora_max) else 0
                )

            # Find best result per task for each TL
            for tl in tl_order:
                if tl == "FE-mlp":
                    continue
                # Get max performing for each tl for this plm out of all layers
                max_metric = task_results[
                    (task_results["Task"] == task)
                    & (task_results["Model Name"] == plm)
                    & (task_results["TL"] == tl)
                ]["Metric"].max()
                results_json[task]["best_models"][plm].append(
                    max_metric if not pd.isna(max_metric) else 0
                )

    # Save update json
    with open("./results/results_matrices.json", "w") as file:
        json.dump(results_json, file)

    # Save text file with the best results
    with open("./results/overleaf/TableScoreboard.tex", "w") as f:
        f.write("\\documentclass{standalone}\n")
        f.write("\\usepackage{graphicx}\n")
        f.write("\\usepackage{multirow}\n")
        f.write("\\begin{document}\n")
        f.write("\\begin{tabular}{c|ccccccc}\n")
        f.write("\\hline\n")
        f.write(
            "\\multirow{2}{*}{\\textbf{Task}} & \\multicolumn{1}{c|}{\\multirow{2}{*}{\\textbf{Score}}} & \\multicolumn{1}{c|}{\\multirow{2}{*}{\\textbf{Metric}}} & \\multicolumn{5}{c}{\\textbf{Best configuration}} \\\\ \\cline{4-8}\n"
        )
        f.write(
            " & \\multicolumn{1}{c|}{} & \\multicolumn{1}{c|}{} & \\textbf{PLM} & \\textbf{TL-method} & \\textbf{Layers used} & \\textbf{Pooling} & \\textbf{Downstream head} \\\\ \\hline\n"
        )

        for result in best_results_per_task:
            f.write(
                f"{result['Task']} & {result['Score']:.4f} & {result['Metric']} & {result['PLM']} & {result['TL-method'].replace('-linear', '').replace('-mlp', '')} & {result['Layers used']} & {result['Pooling']} & {result['Downstream head']} \\\\ \\hline\n"
            )

        f.write("\\end{tabular}\n")
        f.write("\\end{document}")


def write_tables_to_csv(summary_tables, layer_analysis, filename):
    with open(filename, "w") as f:
        for task, table in summary_tables.items():
            f.write(f"Summary table for {task}:\n")
            table.to_csv(f, sep=",")
            f.write("\n")

        for task, tl_tables in layer_analysis.items():
            for tl, table in tl_tables.items():
                f.write(f"Layer analysis for {task} - {tl}:\n")
                table.to_csv(f, sep=",")
                f.write("\n")


def main():
    directory = "results/csv"
    parsed_results = parse_csv_files(directory)

    # # Process the parsed results as needed
    # for result in parsed_results:
    #     print(result)

    summary_tables = generate_summary_table(parsed_results)
    layer_analysis = generate_layer_analysis(parsed_results)
    make_scoreboard(parsed_results)

    # Write the tables to a CSV file
    write_tables_to_csv(
        summary_tables, layer_analysis, "./results/analysis_results.csv"
    )


if __name__ == "__main__":
    main()
