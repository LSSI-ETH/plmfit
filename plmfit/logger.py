import datetime
import os
import json
import matplotlib.pyplot as plt
import torch
import requests
try:
    import env
except ImportError as e:
    raise f"No environment file 'env.py' detected, reverting back to local logger"

class Logger():
    def __init__(self, experiment_name: str, base_dir='loggers', log_to_server=False, server_path=''): 
        self.created_at = datetime.datetime.now()
        self.experiment_name = experiment_name
        formatted_date = self.created_at.strftime("%Y%m%d_%H%M%S")
        self.file_name = f'{formatted_date}_{experiment_name}.log'
        self.base_dir = base_dir
        self.log_to_server = log_to_server

        if log_to_server:
            self.server_url = env.POST_URL
            self.token = env.TOKEN
            self.user = env.USER
            self.server_path = f'/{self.user}/{base_dir}' if server_path == '' else f'/{self.user}/{server_path}'
            self.last_post_time = None  # Track the last time a post was made

        self.ensure_dir(self.base_dir)
        with open(os.path.join(self.base_dir, self.file_name), 'w') as f:
            f.truncate(0)
        self.log(f'#---------Logger initiated with name "{self.experiment_name}" at {self.created_at}---------#')

    def log(self, text: str, force_send=False):
        with open(os.path.join(self.base_dir, self.file_name), 'a') as f:
            f.write(f'{text}\n')

        current_time = datetime.datetime.now()
        # Post to the server if 5 minutes have passed since the last log was successfully posted
        if self.log_to_server and (self.last_post_time is None or (current_time - self.last_post_time).total_seconds() > 300) or force_send:
            self.post_to_server(os.path.join(self.base_dir, self.file_name), self.file_name)

    def ensure_dir(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


    def save_data(self, data, data_name):
        file_path = os.path.join(self.base_dir, f"{self.experiment_name}_data.json")

        # Check if the file already exists and load its contents if it does
        if os.path.exists(file_path):
            with open(file_path, "r") as json_file:
                try:
                    existing_data = json.load(json_file)
                except json.JSONDecodeError:
                    existing_data = {}
        else:
            existing_data = {}

        # Update the existing data with the new data under the specified name
        if data_name in existing_data:
            # If the data_name already exists, update/concatenate the new data
            # This assumes the data under each name is a list
            existing_data[data_name].update(data) if isinstance(data, dict) else existing_data[data_name].extend(data)
        else:
            # If the data_name does not exist, simply add the new data
            existing_data[data_name] = data

        # Write the updated data back to the file
        with open(file_path, "w") as json_file:
            json.dump(existing_data, json_file, indent=4)

        self.log(f'Saved {data_name} with name "{self.experiment_name}_data.json"')

        if self.log_to_server:
            try:
                self.post_to_server(file_path, f"{self.experiment_name}_data.json")
            except Exception as e:
                self.log(f'Error posting data to server: {e}')


    def save_plot(self, plot, plot_name):
        plot_path = os.path.join(self.base_dir, f"{self.experiment_name}_{plot_name}.png")
        plot.savefig(plot_path)
        plt.close(plot)  # Close the plot to free memory
        self.log(f'Saved plot with name "{plot_name}.png"')
        if self.log_to_server:
            try:
                self.post_to_server(plot_path, f"{self.experiment_name}_{plot_name}.png")
            except Exception as e:
                self.log(f'Error posting data to server: {e}')

    def save_model(self, model, model_name):
        plot_path = os.path.join(self.base_dir, f"{self.experiment_name}_{model_name}.pt")
        torch.save(model.state_dict(
        ), plot_path)
        self.log(f'Saved model with name "{self.experiment_name}_{model_name}.pt"')

    def post_to_server(self, file_path, data_name):
        # Ensure the token is included in the headers for authorization
        headers = {'Authorization': f'Bearer {self.token}'}

        try:
            with open(file_path, 'rb') as file:
                files = {'file': (data_name, file)}
                data = {'path': f'{self.server_path}'}
                response = requests.post(self.server_url, files=files, data=data, headers=headers)

            if response.status_code == 200:
                self.last_post_time = datetime.datetime.now()  # Update the last post time
            else:
                self.log(f'Failed to post {data_name} to server. Status code: {response.status_code}')
        except Exception as e:
            self.log(f'Exception occurred while posting to server: {e}')

    def save_log_to_server(self):
        if self.log_to_server:
            self.post_to_server(os.path.join(self.base_dir, self.file_name), self.file_name)