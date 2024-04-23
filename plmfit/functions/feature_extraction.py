from plmfit.shared_utils import utils
import torch
import plmfit.models.downstream_heads as heads
from plmfit.models.fine_tuning import FullRetrainFineTuner
from plmfit.models.hyperparameter_tuner import HyperTuner

def feature_extraction(args, logger):
    # Load dataset
    data = utils.load_dataset(args.data_type)
    
    split = None if args.split is None else data[args.split]
    head_config = utils.load_config(args.head_config)

    ### TODO : Extract embeddings if do not exist
    embeddings = utils.load_embeddings(emb_path=f'{args.output_dir}/extract_embeddings', data_type=args.data_type, model=args.plm, layer=args.layer, reduction=args.reduction)
    assert embeddings != None, "Couldn't find embeddings, you can use extract_embeddings function to create and save the embeddings"

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
            embeddings=embeddings, 
            scores=scores,
            logger=logger,
            split=split,
            experiment_dir=args.experiment_dir
        )

    runner(
        config=head_config,
        embeddings=embeddings, 
        scores=scores,
        logger=logger,
        split=split
    )

def runner(config, embeddings, scores, logger, split=None, on_ray_tuning=False, num_workers=0):
    head_config = config if not on_ray_tuning else utils.adjust_config_to_int(config)

    training_params = head_config['training_parameters']
    data_loaders = utils.create_data_loaders(
            embeddings, scores, scaler=training_params['scaler'], batch_size=training_params['batch_size'], validation_size=training_params['val_split'], split=split, num_workers=num_workers)
     
    if not on_ray_tuning:
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
    final_loss = fine_tuner.train(pred_model, dataloaders_dict=data_loaders, on_ray_tuning=on_ray_tuning)

    return final_loss

def ray_tuning(function_to_run, config, embeddings, scores, logger, experiment_dir, split=None):

    network_type = config['architecture_parameters']['network_type']
    trials = 200 if network_type == 'mlp' else 100
    
    config['training_parameters']['learning_rate'] = (1e-6, 1e-3)
    config['training_parameters']['batch_size'] = (4, 128)
    config['training_parameters']['weight_decay'] = (1e-5, 1e-2)
    if network_type == 'mlp':
        config['architecture_parameters']['hidden_dim'] = (64, 2048)

    tuner = HyperTuner(
        function_to_run=function_to_run, 
        initial_config=config, 
        trials=trials,
        experiment_dir=experiment_dir, 
        embeddings=embeddings, 
        scores=scores,
        logger=logger,
        split=split, 
        on_ray_tuning=True
    )
    
    best_config, best_loss = tuner.fit()

    logger.log(f"Best trial config: {best_config}")
    logger.log(f"Best trial metrics: {best_loss}")

    return best_config
