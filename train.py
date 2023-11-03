from models.pretrained_models import *
from models.downstream_heads import LinearRegression
from models.fine_tuning import *
import argparse
#rom transformers import EsmModel, EsmConfig
from transformers import AutoTokenizer, EsmModel,  EsmForMaskedLM
import esm
import utils

parser = argparse.ArgumentParser(description='Embeddings extraction',fromfile_prefix_chars='@')
parser.add_argument('--layer', type=int , default= 0)
parser.add_argument('--batch_size', type=int , default= 2)
args = parser.parse_args()

if __name__ == '__main__':  
    
    #fine_tuner = FullRetrainFineTuner(epochs = 5 , lr = 0.0006, batch_size = 8,  val_split = 0.2 , log_interval = 1)
    #pro_model = ProGenFamily(progen_model_name = 'progen2-small')
   # 
    #model = ESMFamily(esm_model_name = 'esm_small')
    #configuration = EsmConfig()
    #model = EsmModel(configuration)
    #model.concat_task_specific_head(head)
    #model.fine_tune('aav' ,  fine_tuner, 'two_vs_many_split', 'adam' , 'mse')

    #model.extract_embeddings('aav' , batch_size = args.batch_size , layer = args.layer )
    seq = 'ABCDEFG'
    esm_version  = 'esm2_t30_150M_UR50D'
    model = ESMFamily(esm_version)
    head = LinearRegression(640 , 1)
    model.concat_task_specific_head(head)
    fine_tuner = FullRetrainFineTuner(epochs = 5 , lr = 0.0006, batch_size = 8,  val_split = 0.2 , log_interval = 1)
    model.extract_embeddings('aav' , batch_size = args.batch_size , layer = args.layer )
    print(model.no_parameters)
