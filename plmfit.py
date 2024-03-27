import torch
from plmfit.logger import Logger
import os
import argparse
from plmfit.models.pretrained_models import *
from plmfit.models.fine_tuning import FullRetrainFineTuner, LowRankAdaptationFineTuner
import plmfit.shared_utils.utils as utils
import plmfit.models.downstream_heads as heads
import traceback
import torch.multiprocessing as mp
from ray import tune
from ray.tune import CLIReporter
from ray.train import RunConfig
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.schedulers import ASHAScheduler
import ray
from functools import partial

parser = argparse.ArgumentParser(description='plmfit_args')
# options ['progen2-small', 'progen2-xlarge', 'progen2-oas', 'progen2-medium', 'progen2-base', 'progen2-BFD90' , 'progen2-large']
parser.add_argument('--plm', type=str, default='progen2-small')
parser.add_argument('--ft_method', type=str, default='feature_extraction')
parser.add_argument('--data_type', type=str, default='aav')
# here you specifcy the different splits
parser.add_argument('--data_file_name', type=str, default='data_train')

parser.add_argument('--head_config', type=str, default='linear_head_config.json')
parser.add_argument('--ray_tuning', type=bool, default=False)

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

# Removing the output_dir prefix from experiment_dir
trimmed_experiment_dir = experiment_dir.removeprefix(f"{args.output_dir}/")
logger = Logger(
    experiment_name = args.experiment_name, 
    base_dir= args.experiment_dir, 
    log_to_server=args.logger!='local', 
    server_path=f'{trimmed_experiment_dir}'
)

def init_plm(model_name, logger):
    model = None
    supported_progen2 = ['progen2-small', 'progen2-medium', 'progen2-xlarge']
    supported_ESM = ["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D",
                     "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D"]
    supported_Ankh = ['ankh-base', 'ankh-large', 'ankh2-large']
    supported_Proteinbert = ['proteinbert']

    if 'progen' in model_name:
        assert model_name in supported_progen2, 'Progen version is not supported'
        model = ProGenFamily(model_name, logger)

    elif 'esm' in model_name:
        assert model_name in supported_ESM, 'ESM version is not supported'
        model = ESMFamily(model_name)

    elif 'ankh' in model_name:
        assert model_name in supported_Ankh, 'Ankh version is not supported'
        model = AnkhFamily(model_name)
    elif 'antiberty' in args.plm:
        model = Antiberty()
    elif 'proteinbert' in model_name:
        assert model_name in supported_Proteinbert, 'ProteinBERT version is not supported'
        model = ProteinBERTFamily(logger)
    else:
        raise 'PLM not supported'

    return model

def extract_embeddings(args, logger):
    model = init_plm(args.plm, logger)
    assert model != None, 'Model is not initialized'

    model.extract_embeddings(data_type=args.data_type, layer=args.layer,
                            reduction=args.reduction)

def feature_extraction(config, args, logger, on_ray_tuning=False):
    # Load dataset
    data = utils.load_dataset(args.data_type)
    head_config = config
    # Load embeddings and scores
    ### TODO : Load embeddings if do not exist
    embeddings = utils.load_embeddings(emb_path=f'{args.output_dir}/extract_embeddings', data_type=args.data_type, model=args.plm, layer=args.layer, reduction=args.reduction)
    assert embeddings != None, "Couldn't find embeddings, you can use extract_embeddings function to save {}"

    scores = data['score'].values if head_config['architecture_parameters']['task'] == 'regression' else data['binary_score'].values
    scores = torch.tensor(scores, dtype=torch.float32)

    training_params = head_config['training_parameters']
    data_loaders = utils.create_data_loaders(
            embeddings, scores, scaler=training_params['scaler'], batch_size=training_params['batch_size'], validation_size=training_params['val_split'], num_workers=0)
    
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
    fine_tuner.train(pred_model, dataloaders_dict=data_loaders, on_ray_tuning=on_ray_tuning)

def lora(args, logger):
    # Load dataset
    data = utils.load_dataset(args.data_type)
    
    model = init_plm(args.plm, logger)
    assert model != None, 'Model is not initialized'
    
    head_config = utils.load_config(args.head_config)
    
    logger.save_data(vars(args), 'arguments')
    logger.save_data(head_config, 'head_config')
    # data = data.sample(1000)
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
            encs, scores, scaler=training_params['scaler'], 
            batch_size=training_params['batch_size'], 
            validation_size=training_params['val_split'], 
            dtype=torch.int8, 
            num_workers=2)
    fine_tuner = LowRankAdaptationFineTuner(training_config=training_params, model_name=args.plm, logger=logger)
    model = fine_tuner.set_trainable_parameters(model)
    model.task = pred_model.task
    fine_tuner.train(model, dataloaders_dict=data_loaders)

def full_retrain(args, logger):
    # Load dataset
    data = utils.load_dataset(args.data_type)

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
    
    utils.freeze_parameters(model.py_model)
    utils.set_trainable_parameters(pred_model)
    model.py_model.set_head(pred_model)
    utils.get_parameters(model.py_model, logger=logger)
    data = data.sample(100000)
    encs = model.categorical_encode(data)
    logger.log(model.py_model)
    scores = data['score'].values if head_config['architecture_parameters']['task'] == 'regression' else data['binary_score'].values
    training_params = head_config['training_parameters']
    data_loaders = utils.create_data_loaders(
            encs, scores, scaler=training_params['scaler'], batch_size=training_params['batch_size'], validation_size=training_params['val_split'], dtype=torch.int8)
    fine_tuner = FullRetrainFineTuner(training_config=training_params, logger=logger)
    model.py_model.task = pred_model.task
    fine_tuner.train(model.py_model, dataloaders_dict=data_loaders)

def ray_tuning(head_config, args, logger):
    network_type = head_config['architecture_parameters']['network_type']
    if network_type == 'mlp': 
        head_config['architecture_parameters']['hidden_dim'] = tune.choice([256, 512, 1024, 1536, 2048, 4096])
        head_config['architecture_parameters']['hidden_dropout'] = tune.choice([0.1, 0.25, 0.5, 0.9])
    head_config['training_parameters']['learning_rate'] = tune.loguniform(1e-6, 1e-3)
    head_config['training_parameters']['batch_size'] = tune.choice([8, 16, 32, 64, 128, 256])
    head_config['training_parameters']['weight_decay'] = tune.loguniform(1e-3, 1e-1)

    # Initialize BayesOptSearch
    searcher = BayesOptSearch(
        metric="loss", 
        mode="min"
    )

    reporter = CLIReporter(max_progress_rows=10)

    logger.log("Initializing ray tuning...")
    ray.init(address='auto')

    logger.mute = True # Avoid overpopulating logger with a mixture of training procedures
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(feature_extraction, args=args, logger=logger, on_ray_tuning=True),
            resources={"cpu": 8, "gpu": 2}
        ),
        tune_config=tune.TuneConfig(
            search_alg=searcher,
            num_samples=100,
        ),
        run_config=RunConfig(
            progress_reporter=reporter, 
            log_to_file=(f"ray_stdout.log", "ray_stderr.log"),
            storage_path=f'{experiment_dir}/raytune_results'),
        param_space=head_config,
    )
    results = tuner.fit()
    logger.mute = False # Ok, logger can be normal now

    best_result = results.get_best_result("loss", "min")
    logger.log(f"Best trial config: {best_result.config}")
    logger.log(f"Best trial metrics: {best_result.metrics}")

    return best_result.config


if __name__ == '__main__':

    try:
        if args.function == 'extract_embeddings':
            extract_embeddings(args, logger)
        elif args.function == 'fine_tuning':
            if args.ft_method == 'feature_extraction':
                head_config = utils.load_config(args.head_config)
                if args.ray_tuning:
                    head_config = ray_tuning(head_config, args, logger)
                feature_extraction(head_config, args, logger)
            elif args.ft_method == 'lora':
                lora(args, logger)
            elif args.ft_method == 'full':
                full_retrain(args, logger)
            else:
                raise ValueError('Fine Tuning method not supported')
        else:

            raise ValueError('Function is not supported')
        logger.log("End of process", force_send=True)
    except:
        logger.mute = False
        stack_trace = traceback.format_exc()
        logger.log(stack_trace, force_send=True)
