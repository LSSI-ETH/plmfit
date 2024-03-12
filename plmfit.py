import torch
import plmfit.logger as l
import os
import argparse
from plmfit.models.pretrained_models import *
import plmfit.shared_utils.utils as utils
import plmfit.models.downstream_heads as heads
import traceback
try:
    from env import TOKEN, USER
except ImportError as e:
    print(f"No environment file 'env.py' detected")


parser = argparse.ArgumentParser(description='plmfit_args')
# options ['progen2-small', 'progen2-xlarge', 'progen2-oas', 'progen2-medium', 'progen2-base', 'progen2-BFD90' , 'progen2-large']
parser.add_argument('--plm', type=str, default='progen2-small')
parser.add_argument('--ft_method', type=str, default='feature_extraction')
parser.add_argument('--data_type', type=str, default='aav')
# here you specifcy the different splits
parser.add_argument('--data_file_name', type=str, default='data_train')

parser.add_argument('--head_config', type=str, default='linear_head_config.json')

parser.add_argument('--split', type=str, default='') #TODO implement split logic as well

parser.add_argument('--function', type=str, default='extract_embeddings')
parser.add_argument('--reduction', type=str, default='mean',
                    help='Reduction technique')
parser.add_argument('--layer', type=str, default='last',
                    help='PLM layer to be used')
parser.add_argument('--output_dir', type=str, default='default',
                    help='Output directory for created files')
parser.add_argument('--experiment_name', type=str, default='default',
                    help='Output directory for created files')
parser.add_argument('--experiment_dir', type=str, default='default',
                    help='Output directory for created files')

parser.add_argument('--logger', type=str, default='remote')

args = parser.parse_args()

experiment_dir = args.experiment_dir
if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)

logger = l.Logger(
    experiment_name = args.experiment_name, 
    base_dir= args.experiment_dir, 
    log_to_server=args.logger!='local', 
    server_path=f'{args.function}/{args.experiment_name}')

def init_plm(model_name, logger, task_type='', head=None,):
    model = None
    supported_progen2 = ['progen2-small', 'progen2-medium', 'progen2-xlarge']
    supported_ESM = ["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D",
                     "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D"]
    supported_Ankh = ['ankh-base', 'ankh-large', 'ankh2-large']
    supported_Proteinbert = ['proteinbert']

    if 'progen' in model_name:
        assert model_name in supported_progen2, 'Progen version is not supported'
        if task_type == '':
            model = ProGenFamily(model_name, logger)
        elif task_type == 'classification':
            model = ProGenClassifier(model_name, head)
        else:
            raise f'Task type {task_type} not supported for {model_name}'

    elif 'esm' in model_name:
        assert model_name in supported_ESM, 'ESM version is not supported'
        model = ESMFamily(model_name)

    elif 'ankh' in model_name:
        assert model_name in supported_Ankh, 'Ankh version is not supported'
        model = AnkhFamily(model_name)
    elif 'antiberty' in args.plm:
        model = Antiberty()
    # elif 'proteinbert' in model_name:
    #     assert model_name in supported_Proteinbert, 'ProteinBERT version is not supported'
    #     model = ProteinBERTFamily(model_name)
    else:
        raise 'PLM not supported'

    return model

if __name__ == '__main__':

    try:
        if args.function == 'extract_embeddings':
            
            model = init_plm(args.plm, logger)
            assert model != None, 'Model is not initialized'

            model.extract_embeddings(data_type=args.data_type, layer=args.layer,
                                    reduction=args.reduction)

        elif args.function == 'fine_tuning':
            if args.ft_method == 'feature_extraction':
                # Load dataset
                data = utils.load_dataset(args.data_type)
                # Load embeddings and scores
                ### TODO : Load embeddings if do not exist
                embeddings = utils.load_embeddings(emb_path=f'{args.output_dir}/extract_embeddings/',data_type=args.data_type, model=args.plm, layer=args.layer, reduction=args.reduction)
                assert embeddings != None, "Couldn't find embeddings, you can use extract_embeddings function to save {}"
                
                head_config = utils.load_head_config(args.head_config)

                scores = data['score'].values if head_config['architecture_parameters']['task'] == 'regression' else data['binary_score'].values
                scores = torch.tensor(scores, dtype=torch.float32)
                training_params = head_config['training_parameters']
                data_loaders = utils.create_data_loaders(
                        embeddings, scores, scaler=training_params['scaler'], batch_size=training_params['batch_size'], validation_size=training_params['val_split'])
                
                logger.save_data(vars(args), 'arguments')
                logger.save_data(head_config, 'head_config')

                network_type = head_config['architecture_parameters']['network_type']
                if network_type == 'linear':
                    head_config['architecture_parameters']['input_dim'] = embeddings.shape[1]
                    pred_model = heads.LinearHead(head_config['architecture_parameters'])
                elif network_type == 'mlp':
                    head_config['architecture_parameters']['input_dim'] = embeddings.shape[1]
                    pred_model = heads.MLP(head_config['architecture_parameters'])
                else:
                    raise ValueError('Head type not supported')
                utils.set_trainable_parameters(pred_model)
                fine_tuner = FullRetrainFineTuner(training_config=training_params, logger=logger)
                fine_tuner.train(pred_model, dataloaders_dict=data_loaders)
                
            elif args.ft_method == 'lora':
                
                config = utils.load_head_config(args.head_config)
                if config['network_type'] != args.head:
                    raise f'Wrong configuration file for "{args.head}" head'
                
                base_dir = f'./plmfit/data/'
                output_dir = f'{args.data_type}/models/lora/{args.head}/{args.plm}_{args.layer}_{args.reduction}'
                output_path = base_dir + output_dir
                if (args.logger == 'local'):
                    logger = l.Logger(args.head, base_dir=output_path)
                else:
                    logger = l.ServerLogger(args.head, base_dir=output_path, token=TOKEN, server_path=f'{USER}/{output_dir}')
                
                logger.save_data(vars(args), 'Arguments')
                logger.save_data(config, 'Head config')
                try:
                    data = utils.load_dataset(args.data_type)
                    if args.head == 'logistic_regression':
                        pred_model = heads.LogisticRegression(config)
                        scores = data['binary_score'].values
                        task_type = 'classification'
                    elif args.head == 'linear_regression' or args.head == 'mlp':
                        pred_model = heads.LinearRegression(config) if args.head == 'linear_regression' else heads.MLP(config)
                        scores = data['score'].values
                        task_type = 'regression'
                    
                    model = init_plm(args.plm, task_type=task_type, head=pred_model)
                    assert model != None, 'Model is not initialized'
                    model.set_layer_to_use(args.layer)

                    fine_tuner = LowRankAdaptationFineTuner(
                            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, val_split=0.2, optimizer=args.optimizer, loss_function=args.loss_f, log_interval=100, task_type=task_type)
                    model = fine_tuner.set_trainable_parameters(model)
                    utils.get_parameters(model, logger=logger)
                    scores = torch.tensor(
                        scores, dtype=torch.float32)
                    encs = utils.categorical_encode(data['aa_seq'].values, model.tokenizer, max(data['len'].values), add_bos=True, add_eos=True)
                    data_loaders = utils.create_data_loaders(
                        encs, scores, scaler=args.scaler, batch_size=args.batch_size, dtype=int, validation_size=0.2)
                    fine_tuner.train(
                        model, dataloaders_dict=data_loaders, logger=logger)
                except Exception as e:
                    # Get the entire stack trace as a string
                    stack_trace = traceback.format_exc()
                    logger.log(stack_trace, force_send=True) if args.logger == 'remote' else logger.log(stack_trace)
                if (args.logger == 'remote'):
                        logger.save_log_to_server()
            else:
                raise ValueError('Fine Tuning method not supported')
        else:

            raise ValueError('Function is not supported')
        logger.log("End of process", force_send=True)
    except:
        stack_trace = traceback.format_exc()
        logger.log(stack_trace, force_send=True)
