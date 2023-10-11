import datetime
import os

class Logger():
    file_name = ''
    created_at = None

    def __init__(self, file_name: str):
        
        self.created_at = datetime.datetime.now()
        self.file_name = f'{file_name}_{self.created_at}'
        #file_name = f'{file_name}_{self.created_at}'
        with open(file_name, 'w') as f:
            f.truncate(0)
            f.close()
        
        self.log(f'#---------Logger initiated with name "{file_name}" at {self.created_at}---------#')
            
            
    def log(self, text: str):
        with open(self.file_name, 'a') as f: 
            f.write(text)
            f.write('\n')
            f.close()
            