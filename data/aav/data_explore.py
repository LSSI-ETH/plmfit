import pandas as pd
import utils

if __name__ == '__main__':
    
    data = pd.read_csv('./data/aav/aav_data_full.csv')
    print(data.shape)
    wild_type = utils.get_wild_type('aav')
