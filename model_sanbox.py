from models.pretrained_models import *
from models.models import LinearRegression
import argparse

parser = argparse.ArgumentParser(description='Embeddings extraction',fromfile_prefix_chars='@')
parser.add_argument('--layer', type=int , default= 0)
parser.add_argument('--batch_size', type=int , default= 1)
args = parser.parse_args()

if __name__ == '__main__':  
    model = ProGenPLM()

    #head = LinearRegression(32 , 1)
    #model.concat_task_specific_head(head)
    #model.fine_tune('aav' , 'full_retrain' , epochs = 5 , lr = 0.001 , optimizer = 'adam' , batch_size = 2, train_split_name = 'two_vs_many_split', val_split = 0.2 , loss_f = 'mse' , log_interval = 1)

    model.extract_embeddings('aav' , batch_size = args.batch_size , layer = args.layer )
