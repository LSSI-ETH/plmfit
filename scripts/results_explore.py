import paramiko
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import tempfile
import getpass

def establish_ssh_sftp_sessions(hostname, username, key_path=None):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if key_path and os.path.exists(key_path):
        print("Logging in with private key...")
        private_key = paramiko.Ed25519Key.from_private_key_file(key_path)
        ssh.connect(hostname, username=username, pkey=private_key)
    else:
        print("Private key not found or not specified. Falling back to password authentication...")
        password = getpass.getpass(prompt="Password for SSH connection: ")
        ssh.connect(hostname, username=username, password=password)
    sftp = ssh.open_sftp()
    return ssh, sftp

def find_and_download_json_files(ssh, sftp, base_folder, data_type, task_type, temp_dir):
    # Execute remote find command to get list of matching JSON files
    stdin, stdout, stderr = ssh.exec_command(f"find {base_folder} -type f -name '*{data_type}*{task_type}*_data.json'")
    file_paths = stdout.read().decode().splitlines()
    
    # Download each file to the temporary directory
    for remote_path in file_paths:
        filename = os.path.basename(remote_path)
        local_path = os.path.join(temp_dir, filename)
        sftp.get(remote_path, local_path)
    
    return [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.json')]

def collect_metrics(json_files=None, csv_file=None, data_type='aav', task_type='regression'):
    # # Define the pattern to search for JSON files based on the provided criteria
    # search_pattern = os.path.join(base_folder, f"**/*{data_type}*{task_type}*_data.json")
    
    # # Use glob to find all files matching the pattern, including in subdirectories
    # json_files = glob.glob(search_pattern, recursive=True)
    
    # Initialize a list to store the collected data
    data = []
    
    if json_files is not None:
        # Iterate over the found JSON files and extract the required metrics and details
        for file_path in json_files:
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                metrics = json_data.get('metrics', {})
                
                # Extract details from the filename
                filename = os.path.basename(file_path)
                parts = filename.split('_')
                model_name = parts[1]
                layer = parts[3]
                reduction = parts[4]
                head_type = parts[5]
                
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

                # Combine 'bos' and 'eos' into 'bos/eos' for the 'Reduction' column
                df['Reduction'] = df['Reduction'].replace({'bos': 'bos/eos', 'eos': 'bos/eos'})
                df['Model + Head'] = df['Model Name'] + ' + ' + df['Head Type']
                df['Layer + Reduction'] = df['Layer'] + ' + ' + df['Reduction']
    else:
        df = pd.read_csv(csv_file)
        column_name = "MCC" if task_type == 'classification' else "Spearman"
    

    # Ensure the categorical order
    model_order = ['proteinbert', 'progen2-small', 'progen2-medium', 'progen2-xlarge']
    layer_order = ['first', 'middle', 'last']
    reduction_order = ['mean', 'bos/eos']
    head_order = ['linear', 'mlp']
    
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
    # Main title and subtitle setup
    main_title = f'{data_type.upper()} - {task_type.upper()} Task'
    subtitle = f'{column_name} Across Models, Heads, Layers, and Reduction Methods'

    # Set the main title with more emphasis
    plt.suptitle(main_title, fontsize=20)

    # Set the subtitle with less emphasis and adjust its position
    plt.title(subtitle, fontsize=18, pad=20)

    plt.xticks(rotation=45, ha="right", fontsize=12)  # Adjust font size for x-axis labels
    plt.yticks(rotation=0, fontsize=12)  # Adjust font size for y-axis labels
    plt.xlabel('Model + Head', fontsize=14)  # Explicit x-axis label
    plt.ylabel('Layer + Reduction', fontsize=14)  # Explicit y-axis label
    plt.tight_layout(pad=3)  # Adjust layout to not cut off labels

    plt.show()

    
    # Optionally, save the table to a CSV file
    output_csv = os.path.join(f"{data_type}_{task_type}_metrics_summary.csv")
    df.to_csv(output_csv, index=False)
    print(f"Summary table saved to {output_csv}")

def main():
    use_cache = False
    hostname = 'euler.ethz.ch'
    username = 'estamkopoulo'
    key_path = '/Users/tbikias/Desktop/vaggelis/Config/.ssh/id_ed25519_euler'
    base_folder = '$SCRATCH/fine_tuning/lora'
    data_type = 'meltome'
    task_type = 'regression'
    
    if use_cache:
        collect_metrics(csv_file=f"{data_type}_{task_type}_metrics_summary.csv", data_type=data_type, task_type=task_type)
        return

    ssh, sftp = establish_ssh_sftp_sessions(hostname, username, key_path=key_path)
    
    # Create a temporary directory to store downloaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        json_files = find_and_download_json_files(ssh, sftp, base_folder, data_type, task_type, temp_dir)
        collect_metrics(json_files=json_files, data_type=data_type, task_type=task_type)
    
    # Close SSH and SFTP sessions
    sftp.close()
    ssh.close()

if __name__ == "__main__":
    main()