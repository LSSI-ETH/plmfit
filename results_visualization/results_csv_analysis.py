import os
import glob
import pandas as pd

# Mapping dictionary for task names
task_mapping = {
    'aav_one_vs_many': 'AAV one-vs-rest',
    'aav_sampled': 'AAV sampled',
    'gb1_one_vs_rest': 'GB1 one-vs-rest',
    'gb1_three_vs_rest': 'GB1 three-vs-rest',
    'meltome_mixed': 'Meltome mixed',
    'herH3_one_vs_rest': 'HerH3 one-vs-rest',
    'rbd_one_vs_rest': 'RBD one-vs-rest'
}

task_reverse_mapping = {v: k for k, v in task_mapping.items()}

# List of fine-tuning techniques and their abbreviations
fine_tuning_techniques = {
    'feature_extraction': 'FE',
    'lora_all': 'LoRA',
    'lora_last': 'LoRA-',
    'bottleneck_adapters_all': 'Adapters',
    'bottleneck_adapters_last': 'Adapters-'
}

fine_tuning_techniques_reverse = {v: k for k, v in fine_tuning_techniques.items()}

# Desired order for PLMs and TLs
plm_order = ['proteinbert', 'progen2-small', 'esm2_t33_650M_UR50D',
             'progen2-medium', 'esm2_t36_3B_UR50D', 'progen2-xlarge', 'esm2_t48_15B_UR50D']
tl_order = ['FE', 'LoRA', 'LoRA-', 'Adapters', 'Adapters-']
layer_order = ['first', 'quarter1', 'middle', 'quarter3', 'last']

def parse_csv_files(directory):
    # Find all CSV files in the specified directory
    csv_files = glob.glob(os.path.join(directory, '*.csv'))

    parsed_results = []

    for csv_file in csv_files:
        # Read the CSV file using pandas
        df = pd.read_csv(csv_file)

        # Check if the CSV file has the expected columns
        expected_columns = ['File', 'Model Name', 'Layer', 'Reduction',
                            'Head Type', 'Model + Head', 'Layer + Reduction']
        if all(column in df.columns for column in expected_columns) and ('Spearman' in df.columns or 'MCC' in df.columns):
            # Parse the task name from the 'File' column
            def get_task(file_name):
                for key in task_mapping:
                    if key in file_name:
                        return task_mapping[key]
                return 'Unknown'
            
            # Parse the fine-tuning technique from the file name
            def get_fine_tuning_technique(file_name):
                for technique in fine_tuning_techniques:
                    if technique in file_name:
                        return fine_tuning_techniques[technique]
                return 'Unknown'
            df.rename(columns={'Spearman': 'Metric', 'MCC': 'Metric'}, inplace=True)
            df['Task'] = df['File'].apply(get_task)
            df['TL'] = get_fine_tuning_technique(os.path.basename(csv_file))
            parsed_results.append(df)
        else:
            print(
                f"Skipping file {csv_file} as it does not have the expected columns")

    return parsed_results


def generate_summary_table(parsed_results):
    summary_tables = {}

    for task in task_mapping.values():
        task_results = [df[df['Task'] == task] for df in parsed_results]
        task_results = pd.concat(task_results)

        summary_table = task_results.groupby(['Model Name', 'TL'])[
            'Metric'].max().unstack().fillna(0)
        
        # Reorder the PLMs and TLs
        summary_table = summary_table.reindex(
            index=plm_order, columns=tl_order)
        summary_tables[task] = summary_table

    return summary_tables


def generate_layer_analysis(parsed_results):
    layer_analysis = {}

    for task in task_mapping.values():
        task_results = [df[df['Task'] == task] for df in parsed_results]
        task_results = pd.concat(task_results)

        for tl in tl_order:
            tl_results = task_results[task_results['TL'] == tl]
            layer_table = tl_results.groupby(['Model Name', 'Layer'])[
                'Metric'].max().unstack().fillna(0)

            # Reorder the PLMs and layers
            layer_table = layer_table.reindex(
                index=plm_order, columns=layer_order)
            if task not in layer_analysis:
                layer_analysis[task] = {}
            layer_analysis[task][tl] = layer_table

    return layer_analysis

# find the best model and layer combo for each task and for each TL and print the model with the corresponding layer, the performance and the info
def find_best_model(parsed_results):
    # load csv 'combined_results.csv' which has gpu and cpu information for each experiment
    combined_results = pd.read_csv('./results_visualization/combined_results.csv')

    for task in task_mapping.values():
        task_results = [df[df['Task'] == task] for df in parsed_results]
        task_results = pd.concat(task_results)

        for tl in tl_order:
            tl_results = task_results[task_results['TL'] == tl]
            max_metric = tl_results['Metric'].max()
            best_model = tl_results[tl_results['Metric'] == max_metric]['Model Name'].values[0]
            best_layer = tl_results[tl_results['Metric'] == max_metric]['Layer'].values[0]
            
            # Find the max GPU usage for the best model and layer combo
            gpu_usage_row = combined_results[
                (combined_results['great_grandparent_folder'] == fine_tuning_techniques_reverse[tl]) &
                (
                    combined_results['grandparent_folder'].str.contains(f'{task_reverse_mapping[task]}_{best_model}_bottleneck_adapters_{best_layer}') |
                    combined_results['grandparent_folder'].str.contains(f'{task_reverse_mapping[task]}_{best_model}_lora_{best_layer}') |
                    combined_results['grandparent_folder'].str.contains(
                        f'{task_reverse_mapping[task]}_{best_model}_feature_extraction_{best_layer}')
                )
            ]

            if not gpu_usage_row.empty:
                gpu_usage = gpu_usage_row['gpu_max_usage'].values[0]
                print(
                    f"Best model for {task} with {tl} is {best_model} with layer {best_layer} --- a performance of {max_metric} and a Max GPU usage of {gpu_usage}")
            else:
                print(
                    f"Best model for {task} with {tl} is {best_model} with layer {best_layer} --- a performance of {max_metric}")

def write_tables_to_csv(summary_tables, layer_analysis, filename):
    with open(filename, 'w') as f:
        for task, table in summary_tables.items():
            f.write(f"Summary table for {task}:\n")
            table.to_csv(f, sep=',')
            f.write('\n')

        for task, tl_tables in layer_analysis.items():
            for tl, table in tl_tables.items():
                f.write(f"Layer analysis for {task} - {tl}:\n")
                table.to_csv(f, sep=',')
                f.write('\n')

def main():
    directory = 'results/csv'
    parsed_results = parse_csv_files(directory)

    # # Process the parsed results as needed
    # for result in parsed_results:
    #     print(result)

    summary_tables = generate_summary_table(parsed_results)
    layer_analysis = generate_layer_analysis(parsed_results)
    find_best_model(parsed_results)
    
    # Write the tables to a CSV file
    write_tables_to_csv(summary_tables, layer_analysis, './results/analysis_results.csv')

if __name__ == "__main__":
    main()
