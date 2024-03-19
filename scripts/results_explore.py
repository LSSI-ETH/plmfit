import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def collect_metrics(base_folder, data_type, task_type):
    # Define the pattern to search for JSON files based on the provided criteria
    search_pattern = os.path.join(base_folder, f"**/*{data_type}*{task_type}*_data.json")
    
    # Use glob to find all files matching the pattern, including in subdirectories
    json_files = glob.glob(search_pattern, recursive=True)
    
    # Initialize a list to store the collected data
    data = []
    
    # Iterate over the found JSON files and extract the required metrics and details
    for file_path in json_files:
        with open(file_path, 'r') as file:
            json_data = json.load(file)
            metrics = json_data.get('metrics', {})
            
            # Extract details from the filename
            filename = os.path.basename(file_path)
            parts = filename.split('_')
            model_name = parts[1]
            layer = parts[4]
            reduction = parts[5]
            head_type = parts[6]
            
            # Depending on the task type, extract the relevant metrics
            entry = {
                'File': filename,
                'Model Name': model_name,
                'Layer': layer,
                'Reduction': reduction,
                'Head Type': head_type
            }
            if task_type == 'classification':
                entry['Accuracy'] = metrics.get('accuracy', None)
                entry['MCC'] = metrics.get('mcc', None)
                column_name = "MCC"
            elif task_type == 'regression':
                entry['RMSE'] = metrics.get('rmse', None)
                entry['Spearman'] = metrics.get('spearman', None)
                column_name = "Spearman"
            data.append(entry)
    
    # Convert the collected data into a pandas DataFrame for easy tabular representation
    df = pd.DataFrame(data)

    # Ensure the categorical order
    model_order = ['progen2-small', 'progen2-medium', 'progen2-xlarge']
    layer_order = ['first', 'middle', 'last']
    reduction_order = ['mean', 'eos']
    head_order = ['linear', 'mlp']

    # Step 1: Create concatenated columns first
    df['Model + Head'] = df['Model Name'] + ' + ' + df['Head Type']
    df['Layer + Reduction'] = df['Layer'] + ' + ' + df['Reduction']

    # Pivot without sorting
    heatmap_data = df.pivot(index="Layer + Reduction", columns="Model + Head", values=column_name)

    # Step 2: Reorder the pivoted DataFrame directly
    # Define the desired order for the axes
    model_head_order = [m + ' + ' + h for m in model_order for h in head_order]
    layer_reduction_order = [l + ' + ' + r for l in layer_order for r in reduction_order]

    # Reindex the pivoted DataFrame to enforce the order
    heatmap_data = heatmap_data.reindex(index=layer_reduction_order, columns=model_head_order)

    # Plotting the heatmap with the color scale adjusted from -1 to 1
    plt.figure(figsize=(14, 10))  # Slightly larger figure size for better readability
    sns.set(font_scale=1.2)  # Adjust font scale for better readability
    ax = sns.heatmap(heatmap_data, annot=True, cmap="RdBu_r", fmt=".2f", linewidths=.5,
                    cbar_kws={'label': column_name}, center=0, vmin=-1, vmax=1)
    # Setting vmin and vmax for the color scale

    plt.title(f'{column_name} Across Models, Heads, Layers, and Reduction Methods', fontsize=18, pad=20)
    plt.xticks(rotation=45, ha="right", fontsize=12)  # Adjust font size for x-axis labels
    plt.yticks(rotation=0, fontsize=12)  # Adjust font size for y-axis labels
    plt.xlabel('Model + Head', fontsize=14)  # Explicit x-axis label
    plt.ylabel('Layer + Reduction', fontsize=14)  # Explicit y-axis label
    plt.tight_layout(pad=3)  # Adjust layout to not cut off labels

    plt.show()

    
    # Optionally, save the table to a CSV file
    output_csv = os.path.join(base_folder, f"{data_type}_{task_type}_metrics_summary.csv")
    df.to_csv(output_csv, index=False)
    print(f"Summary table saved to {output_csv}")

base_folder = '/Users/tbikias/Desktop/vaggelis/Downloads/fine_tuning/'
data_type = 'meltome'
task_type = 'regression'  # or 'classification'
collect_metrics(base_folder, data_type, task_type)
