from plmfit.shared_utils import utils
import torch
import plmfit.models.downstream_heads as heads
from plmfit.models.fine_tuners import FullRetrainFineTuner
from plmfit.models.hyperparameter_tuner import HyperTuner
def blosum(args, logger):
    # Load dataset
    data = utils.load_dataset(args.data_type)
    # data = data[:10000]
    # This checks if args.split is set to 'sampled' and if 'sampled' is not in data, or if args.split is not a key in data.
    split = None if args.split == 'sampled' and 'sampled' not in data else data.get(args.split)
    head_config = utils.load_config(f'training/{args.head_config}')
    
    logger.log("Initializing BLOSUM62 encoding...")
    max_len = data['len'].max()
    encs = utils.blosum62_encode(data['aa_seq'].values, pad_to_length=max_len, logger=logger)
    encs = encs.reshape(encs.shape[0], -1)
    logger.log(f"BLOSUM62 encoding completed!\nEncoded sequences shape: {encs.shape}")

    if head_config['architecture_parameters']['task'] == 'regression':
        scores = data['score'].values 
    elif head_config['architecture_parameters']['task'] == 'classification':
        scores = data['binary_score'].values
    # TODO: Make scores data type agnostic
    elif "multilabel" in head_config['architecture_parameters']['task']:
        scores = data[["mouse","cattle","bat"]].values
    else:
        raise f"Task type {head_config['architecture_parameters']['task']} not supported."
    scores = torch.tensor(scores, dtype=torch.float32)
    
    logger.save_data(vars(args), 'arguments')

    if args.ray_tuning == 'True': 
        head_config = ray_tuning(
            runner, 
            config=head_config,
            encodings=encs, 
            scores=scores,
            logger=logger,
            split=split,
            experiment_dir=args.experiment_dir
        )

    runner(
        config=head_config,
        encodings=encs, 
        scores=scores,
        logger=logger,
        split=split
    )

def runner(config, encodings, scores, logger, split=None, on_ray_tuning=False, num_workers=0):
    head_config = config if not on_ray_tuning else utils.adjust_config_to_int(config)

    training_params = head_config['training_parameters']
    data_loaders = utils.create_data_loaders(
            encodings, scores, scaler=training_params['scaler'], batch_size=training_params['batch_size'], validation_size=training_params['val_split'], split=split, num_workers=num_workers)
    
    if not on_ray_tuning:
        logger.save_data(head_config, 'head_config')

    network_type = head_config['architecture_parameters']['network_type']
    if network_type == 'linear':
        head_config['architecture_parameters']['input_dim'] = encodings.shape[1]
        pred_model = heads.LinearHead(head_config['architecture_parameters'])
    elif network_type == 'mlp':
        head_config['architecture_parameters']['input_dim'] = encodings.shape[1]
        pred_model = heads.MLP(head_config['architecture_parameters'])
    else:
        raise ValueError('Head type not supported')
    
    utils.set_trainable_parameters(pred_model)
    fine_tuner = FullRetrainFineTuner(training_config=training_params, logger=logger)
    final_loss = fine_tuner.train(pred_model, dataloaders_dict=data_loaders, on_ray_tuning=on_ray_tuning, blosum=True)

    return final_loss

def ray_tuning(function_to_run, config, encodings, scores, logger, experiment_dir, split=None):
    network_type = config['architecture_parameters']['network_type']
    trials = 200 if network_type == 'mlp' else 100
    
    config['training_parameters']['learning_rate'] = (1e-6, 1e-3)
    config['training_parameters']['batch_size'] = (4, 128)
    config['training_parameters']['weight_decay'] = (1e-5, 1e-2)
    if network_type == 'mlp':
        config['architecture_parameters']['hidden_dim'] = (4, 1024)

    tuner = HyperTuner(
        function_to_run=function_to_run, 
        initial_config=config, 
        trials=trials,
        experiment_dir=experiment_dir, 
        encodings=encodings, 
        scores=scores,
        logger=logger,
        split=split, 
        on_ray_tuning=True
    )

    best_config, best_loss = tuner.fit()

    logger.log(f"Best trial config: {best_config}")
    logger.log(f"Best trial metrics: {best_loss}")

    return best_config