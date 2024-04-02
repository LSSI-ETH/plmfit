import torch
import plmfit.logger as l
import os
import argparse
from plmfit.models.pretrained_models import *
import plmfit.shared_utils.utils as utils
import plmfit.models.downstream_heads as heads
import traceback


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
    os.makedirs(experiment_dir, exist_ok=True)

logger = l.Logger(
    experiment_name = args.experiment_name, 
    base_dir= args.experiment_dir, 
    log_to_server=args.logger!='local', 
    server_path=f'{args.function}/{args.experiment_name}')

def init_plm(model_name, logger):
    model = None
    supported_progen2 = ['progen2-small', 'progen2-medium', 'progen2-xlarge']
    supported_ESM = ["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D",
                     "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D","esm2_t36_3B_UR50D"]
    supported_Ankh = ['ankh-base', 'ankh-large', 'ankh2-large']
    supported_Proteinbert = ['proteinbert']

    if 'progen' in model_name:
        assert model_name in supported_progen2, 'Progen version is not supported'
        model = ProGenFamily(model_name, logger)

    elif 'esm' in model_name:
        assert model_name in supported_ESM, 'ESM version is not supported'
        model = ESMFamily(model_name,logger,args.function)

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

            # Load dataset
            data = utils.load_dataset(args.data_type)

            if args.ft_method == 'feature_extraction':
                # Load embeddings and scores
                ### TODO : Load embeddings if do not exist
                embeddings = utils.load_embeddings(emb_path=f'{args.output_dir}/extract_embeddings/',data_type=args.data_type, model=args.plm, layer=args.layer, reduction=args.reduction)
                assert embeddings != None, "Couldn't find embeddings, you can use extract_embeddings function to save {}"
                
                head_config = utils.load_config(args.head_config)

                split = None
                training_params = head_config['training_parameters']
                if "multilabel" in head_config['architecture_parameters']['task']:
                    # TODO : Make multilabel task agnostic
                    scores = data[["mouse","cattle","bat"]].values
                    scores_dict = {0:"mouse",1:"cattle",2:"bat"}
                    split = data["random"].values
                else:
                    scores = data['score'].values if head_config['architecture_parameters']['task'] == 'regression' else data['binary_score'].values
                    scores = torch.tensor(scores, dtype=torch.float32)

                data_loaders = utils.create_data_loaders(
                        embeddings, scores, split = split, scaler=training_params['scaler'], batch_size=training_params['batch_size'], validation_size=training_params['val_split'])

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

                model = init_plm(args.plm, logger)
                assert model != None, 'Model is not initialized'
                
                head_config = utils.load_config(args.head_config)
                
                logger.save_data(vars(args), 'arguments')
                logger.save_data(head_config, 'head_config')

                network_type = head_config['architecture_parameters']['network_type']
                if network_type == 'linear':
                    head_config['architecture_parameters']['input_dim'] = model.emb_layers_dim
                    pred_model = heads.LinearHead(head_config['architecture_parameters'])
                elif network_type == 'mlp':
                    head_config['architecture_parameters']['input_dim'] = model.emb_layers_dim
                    pred_model = heads.MLP(head_config['architecture_parameters'])
                else:
                    raise ValueError('Head type not supported')
                
                model.py_model.set_head(pred_model)
                model.py_model.reduction = args.reduction
                model.set_layer_to_use(args.layer)
                model.py_model.layer_to_use = model.layer_to_use
                encs = model.categorical_encode(data)


                scores = data['score'].values if head_config['architecture_parameters']['task'] == 'regression' else data['binary_score'].values
                training_params = head_config['training_parameters']
                data_loaders = utils.create_data_loaders(
                        encs, scores, scaler=training_params['scaler'], batch_size=training_params['batch_size'], validation_size=training_params['val_split'])
                
                fine_tuner = LowRankAdaptationFineTuner(training_config=training_params, logger=logger)
                model = fine_tuner.set_trainable_parameters(model)
                model.task = pred_model.task
                fine_tuner.train(model, dataloaders_dict=data_loaders)
            
            elif args.ft_method == 'adapter':
                logger.log("Finetuning method is adapters")
                model = init_plm(args.plm, logger)
                assert model != None, 'Model is not initialized'

                head_config = utils.load_config(args.head_config)
                training_params = head_config['training_parameters']
                fine_tuner = AdapterFineTuner(training_config=training_params, logger=logger)
                logger.log("Finetuner class is initialized")

                logger.save_data(vars(args), 'arguments')
                logger.save_data(head_config, 'head_config')

                network_type = head_config['architecture_parameters']['network_type']
                if network_type == 'linear':
                    head_config['architecture_parameters']['input_dim'] = model.emb_layers_dim
                    pred_model = heads.LinearHead(head_config['architecture_parameters'])
                elif network_type == 'mlp':
                    head_config['architecture_parameters']['input_dim'] = model.emb_layers_dim
                    pred_model = heads.MLP(head_config['architecture_parameters'])
                else:
                    raise ValueError('Head type not supported')
                logger.log("Prediction head is initialized")

                model = fine_tuner.add_adapter(model)
                logger.log("Adapters are added.")
                logger.log(model.adapters)
                model = fine_tuner.set_trainable_parameters(model)
                logger.log("Base model is frozen, only adapters are trainable...")
                model.task = pred_model.task
                fine_tuner.set_head(model,pred_model)
                logger.log("Prediction head of the model is set.")
                encs = model.categorical_encode(data['aa_seq'].values, model.tokenizer,max(data['len'].values))
                logger.log("Encoding of the sequences is complete!")

                split = None
                training_params = head_config['training_parameters']
                if "multilabel" in head_config['architecture_parameters']['task']:
                    # TODO : Make multilabel task agnostic
                    scores = data[["mouse","cattle","bat"]].values
                    # TODO : Do something with the scores_dict
                    scores_dict = {0:"mouse",1:"cattle",2:"bat"}
                    split = data["random"].values
                else:
                    scores = data['score'].values if head_config['architecture_parameters']['task'] == 'regression' else data['binary_score'].values
                    scores = torch.tensor(scores, dtype=torch.float32)

                data_loaders = utils.create_data_loaders(
                        encs, scores, split = split, scaler=training_params['scaler'], batch_size=training_params['batch_size'], validation_size=training_params['val_split'],dtype = torch.int64)
                
                logger.log("Dataloaders are created. Starting training...")
                fine_tuner.train(model, dataloaders_dict=data_loaders)
            else:
                raise ValueError('Fine Tuning method not supported')
        else:

            raise ValueError('Function is not supported')
        logger.log("End of process", force_send=True)
    except:
        stack_trace = traceback.format_exc()
        logger.log(stack_trace, force_send=True)
        