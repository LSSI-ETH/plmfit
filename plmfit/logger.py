import datetime
import os


class Logger():
    def __init__(self, file_name: str):
        self.created_at = datetime.datetime.now()
        formatted_date = self.created_at.strftime("%Y%m%d_%H%M%S")
        self.file_name = f'{file_name}_{formatted_date}.log'

        # Set the location relative to the root of the project
        self.location = os.path.join(
            os.path.dirname(__file__), '..', 'loggers')

        # Create the directory if it doesn't exist
        if not os.path.exists(self.location):
            os.makedirs(self.location)

        # Initialize the log file
        with open(os.path.join(self.location, self.file_name), 'w') as f:
            f.truncate(0)

        self.log(
            f'#---------Logger initiated with name "{file_name}" at {self.created_at}---------#')

    def log(self, text: str):
        with open(os.path.join(self.location, self.file_name), 'a') as f:
            f.write(text + '\n')
