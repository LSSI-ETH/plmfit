import torch
import plmfit.logger as l
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
parser.add_argument('--embs', default=None)

# option ['mlp' , 'cnn' , 'inception', '{a custome head}' , 'attention']
parser.add_argument('--head', type=str, default='linear')
parser.add_argument('--head_config', type=str, default='config_mlp')
parser.add_argument('--task', type=str, default='cls')

parser.add_argument('--gpus', type=int, default=0)
parser.add_argument('--gres', type=str, default='gpumem:24g')
parser.add_argument('--mem-per-cpu', type=int, default=0)
parser.add_argument('--nodes', type=int, default=1)


parser.add_argument('--training_split', type=str, default='two_vs_many_split')
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--val_split', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--scaler', type=str, default=None)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--loss_f', type=str, default='mse')
parser.add_argument('--function', type=str, default='extract_embeddings')
parser.add_argument('--reduction', type=str, default='mean',
                    help='Reduction technique')
parser.add_argument('--layer', type=str, default='last',
                    help='PLM layer to be used')

parser.add_argument('--output_dir', type=str, default='default',
                    help='Output directory for created files')

parser.add_argument('--logger', type=str, default='local')

args = parser.parse_args()


def init_plm(model_name, task_type='', head=None):
    model = None
    supported_progen2 = ['progen2-small', 'progen2-medium', 'progen2-xlarge']
    supported_ESM = ["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D",
                     "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D"]
    supported_Ankh = ['ankh-base', 'ankh-large', 'ankh2-large']
    supported_Proteinbert = ['proteinbert']
    supported_Proteinbert = ['proteinbert']

    if 'progen' in model_name:
        assert model_name in supported_progen2, 'Progen version is not supported'
        if task_type == '':
            model = ProGenFamily(model_name)
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

    if args.function == 'extract_embeddings':
        model = init_plm(args.plm)
        assert model != None, 'Model is not initialized'

        model.extract_embeddings(data_type=args.data_type, layer=args.layer,
                                 reduction=args.reduction, output_dir=args.output_dir)

    elif args.function == 'fine_tuning':
        if args.ft_method == 'feature_extraction':

            config = utils.load_head_config(args.head_config)
            if config['network_type'] != args.head:
                raise f'Wrong configuration file for "{args.head}" head'

            # Load dataset
            data = utils.load_dataset(args.data_type)

            # Load embeddings and scores
            embeddings = utils.load_embeddings(emb_path=args.embs,
                                               data_type=args.data_type, model=args.plm, layer=args.layer, reduction=args.reduction)

            if args.head == 'logistic_regression':
                binary_scores = data['binary_score'].values
                binary_scores = torch.tensor(
                    binary_scores, dtype=torch.float32)

                data_loaders = utils.create_data_loaders(
                    embeddings, binary_scores, scaler=args.scaler, batch_size=args.batch_size)

                pred_model = heads.LogisticRegression(config)
                
                base_dir = f'./plmfit/data/'
                output_dir = f'{args.data_type}/models/regression/{args.head}/{args.plm}_{args.layer}_{args.reduction}'
                output_path = base_dir + output_dir
                if (args.logger == 'local'):
                    logger = l.Logger(args.head, base_dir=output_path)
                else:
                    logger = l.ServerLogger(args.head, base_dir=output_path, token=TOKEN, server_path=f'{USER}/{output_dir}')
                
                logger.save_data(vars(args), 'Arguments')
                logger.save_data(config, 'Head config')

                fine_tuner = FullRetrainFineTuner(
                    epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, val_split=0.2, optimizer=args.optimizer, loss_function=args.loss_f, log_interval=-1, task_type='classification')
                fine_tuner.train(
                    pred_model, dataloaders_dict=data_loaders, logger=logger)
                logger.save_log_to_server()
            elif args.head == 'linear_regression' or args.head == 'mlp':
                scores = data['score'].values
                scores = torch.tensor(
                    scores, dtype=torch.float32)

                data_loaders = utils.create_data_loaders(
                    embeddings, scores, scaler=args.scaler, batch_size=args.batch_size)

                pred_model = heads.LinearRegression(config) if args.head == 'linear_regression' else heads.MLP(config)
                
                base_dir = f'./plmfit/data/'
                output_dir = f'{args.data_type}/models/regression/{args.head}/{args.plm}_{args.layer}_{args.reduction}'
                output_path = base_dir + output_dir
                if (args.logger == 'local'):
                    logger = l.Logger(args.head, base_dir=output_path)
                else:
                    logger = l.ServerLogger(args.head, base_dir=output_path, token=TOKEN, server_path=f'{USER}/{output_dir}')
                
                logger.save_data(vars(args), 'Arguments')
                logger.save_data(config, 'Head config')
                fine_tuner = FullRetrainFineTuner(
                    epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, val_split=0.2, optimizer=args.optimizer, loss_function=args.loss_f, log_interval=-1, task_type='regression')
                fine_tuner.train(
                    pred_model, dataloaders_dict=data_loaders, logger=logger)
                logger.save_log_to_server()
            else:
                raise ValueError('Head type not supported')
            
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
            logger.save_log_to_server()
        else:
            raise ValueError('Fine Tuning method not supported')
    else:

        raise ValueError('Function is not supported')
