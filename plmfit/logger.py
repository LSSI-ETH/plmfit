import datetime
import os
import json
import matplotlib.pyplot as plt
import torch


class Logger():
    def __init__(self, experiment_name: str, base_dir = './loggers'):
        self.created_at = datetime.datetime.now()
        self.experiment_name = experiment_name
        formatted_date = self.created_at.strftime("%Y%m%d_%H%M%S")
        self.file_name = f'{formatted_date}_{experiment_name}.log'

        # Set the location relative to the root of the project
        self.base_dir = base_dir

        self.ensure_dir(self.base_dir)

        # Initialize the log file
        with open(os.path.join(self.base_dir, self.file_name), 'w') as f:
            f.truncate(0)

        self.log(
            f'#---------Logger initiated with name "{self.experiment_name}" at {self.created_at}---------#')

    def log(self, text: str):
        with open(os.path.join(self.base_dir, self.file_name), 'a') as f:
            f.write(f'{text}\n')

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

    def save_plot(self, plot, plot_name):
        plot_path = os.path.join(self.base_dir, f"{self.experiment_name}_{plot_name}.png")
        plot.savefig(plot_path)
        plt.close(plot)  # Close the plot to free memory
        self.log(f'Saved plot with name "{plot_name}.png"')

    def save_model(self, model, model_name):
        plot_path = os.path.join(self.base_dir, f"{self.experiment_name}_{model_name}.pt")
        torch.save(model.state_dict(
        ), plot_path)
        self.log(f'Saved model with name "{self.experiment_name}_{model_name}.pt"')