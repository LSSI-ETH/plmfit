import argparse 
from models.pretrained_models import ESMFamily, ProGenFamily
from models.downstream_heads import LinearRegression
from models.fine_tuning import FullRetrainFineTuner

parser = argparse.ArgumentParser(description='plmfit_args')
parser.add_argument('--model_name', type=str, default='esm2_t30_150M_UR50D') ## options ['progen2-small', 'progen2-xlarge', 'progen2-oas', 'progen2-medium', 'progen2-base', 'progen2-BFD90' , 'progen2-large']
parser.add_argument('--do', type=str, default = 'fine-tuning') ### options ['fine_tuning' , 'extract_embeddings']
parser.add_argument('--ft_method', type=str , default='feature_extraction')
parser.add_argument('--data_type', type=str , default= 'aav')
parser.add_argument('--layer', type=int , default= 0)
parser.add_argument('--embs', type=str , default= 'aav_progen2-small_embs_layer12_mean')
parser.add_argument('--head', type=str , default= 'linear') # option ['mlp' , 'cnn' , 'inception', '{a custome head}' , 'attention']
parser.add_argument('--head_config', type=str , default= 'config_mlp')
parser.add_argument('--task', type=str , default= 'cls')

parser.add_argument('--gpus', type=int , default=0) 
parser.add_argument('--gres', type=str , default='gpumem:24g') 
parser.add_argument('--mem-per-cpu', type=int , default= 0)
parser.add_argument('--nodes', type=int , default= 1)



parser.add_argument('--training_split', type=str , default='two_vs_many_split') 
parser.add_argument('--batch_size', type=int , default= 5)
parser.add_argument('--epochs', type=int , default= 5)
parser.add_argument('--val_split', type=float , default= 0.2)
parser.add_argument('--lr', type=float , default= 0.0001)
parser.add_argument('--optimizer', type=str , default='adam') 



args = parser.parse_args()  

model = None
fine_tuner = None

if __name__=='__main__':
    print('plmfit')
    
    if 'progen' in args.model_name:
        print('do something with progen family') ##### progen2-small
        model = ProGenFamily(args.model_name)
        
    elif 'esm' in args.model_name:
        print('do something with esm family') ##### 'esm2_t30_150M_UR50D'
        model = ESMFamily(args.model_name)
        
    else:
        raise NameError('Not supported protein Language Model')
        
        
    if args.do == 'fine-tuning':
        print(f'fine-tuning {model.version}')
        head = LinearRegression(33 , 1)
        model.concat_task_specific_head(head)
        
        if args.ft_method == 'full-retrain':
            fine_tuner = FullRetrainFineTuner(epochs = args.epochs , lr = args.lr , batch_size = args.batch_size ,  val_split = args.val_split )
        else:
            NameError('Not supported fine tuning method')
            
        model.fine_tune(args.data_type ,  fine_tuner, 'two_vs_many_split', 'adam' , 'mse')

            
    elif args.do == 'extract_embeddings':
        print(f'extracting embedding for {model.version}')
        model.extract_embeddings(args.data_type , batch_size = args.batch_size , layer = args.layer )
        
    else:
        raise NameError('Not supported plmfit function')
        
    
    
    
    
        
        
        
        
    
    
    
    
    
    
    
