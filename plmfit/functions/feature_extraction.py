from plmfit.shared_utils import utils
import torch
import plmfit.models.downstream_heads as heads
from plmfit.models.fine_tuners import FullRetrainFineTuner
from plmfit.models.hyperparameter_tuner import HyperTuner

def feature_extraction(args, logger):
    # Load dataset
    data = utils.load_dataset(args.data_type)
    
    # This checks if args.split is set to 'sampled' and if 'sampled' is not in data, or if args.split is not a key in data.
    split = None if args.split == 'sampled' and 'sampled' not in data else data.get(args.split)
    head_config = utils.load_config(args.head_config)
    weights = None if args.weights is None else data.get(args.weights)

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
            weights=weights,
            experiment_dir=args.experiment_dir
        )

    runner(
        config=head_config,
        embeddings=embeddings, 
        scores=scores,
        logger=logger,
        split=split,
        weights=weights
    )

# def feature_extraction_lightning(config, args, logger, on_ray_tuning=False):
#     # Load dataset
#     data = utils.load_dataset(args.data_type)
#     head_config = config if not on_ray_tuning else utils.adjust_config_to_int(config)
#     split = None if args.split is None else data[args.split]
    
#     # Load embeddings and scores
#     ### TODO : Load embeddings if do not exist
#     embeddings = utils.load_embeddings(emb_path=f'{args.output_dir}/extract_embeddings', data_type=args.data_type, model=args.plm, layer=args.layer, reduction=args.reduction)
#     assert embeddings != None, "Couldn't find embeddings, you can use extract_embeddings function to save {}"

#     scores = data['score'].values if head_config['architecture_parameters']['task'] == 'regression' else data['binary_score'].values
#     scores = torch.tensor(scores, dtype=torch.float32)

#     training_params = head_config['training_parameters']
#     data_loaders = utils.create_data_loaders(
#             embeddings, scores, scaler=training_params['scaler'], batch_size=training_params['batch_size'], validation_size=training_params['val_split'], split=split, num_workers=NUM_WORKERS)
    
#     logger.save_data(vars(args), 'arguments')
#     logger.save_data(head_config, 'head_config')

#     network_type = head_config['architecture_parameters']['network_type']
#     if network_type == 'linear':
#         head_config['architecture_parameters']['input_dim'] = embeddings.shape[1]
#         pred_model = heads.LinearHead(head_config['architecture_parameters'])
#     elif network_type == 'mlp':
#         head_config['architecture_parameters']['input_dim'] = embeddings.shape[1]
#         pred_model = heads.MLP(head_config['architecture_parameters'])
#     else:
#         raise ValueError('Head type not supported')
    
#     utils.set_trainable_parameters(pred_model)
    
#     model = LightningModel(pred_model, head_config['training_parameters'], plmfit_logger=logger, log_interval=100)
#     lightning_logger = TensorBoardLogger(save_dir=logger.base_dir, version=0, name="lightning_logs")

#     callbacks = [DeviceStatsMonitor(True)]
#     if model.early_stopping is not None: callbacks.append(model.early_stopping)
#     if on_ray_tuning:
#         ray_callback = TuneReportCallback(
#             {
#                 "loss": "val_loss"
#             },
#             on="validation_end"
#         )
#         callbacks.append(ray_callback)


#     trainer = L.Trainer(
#         default_root_dir=logger.base_dir,
#         logger=lightning_logger, 
#         max_epochs=model.hparams.epochs, 
#         enable_progress_bar=False, 
#         accumulate_grad_batches=model.gradient_accumulation_steps(),
#         gradient_clip_val=model.gradient_clipping(),
#         limit_train_batches=model.epoch_sizing(),
#         limit_val_batches=model.epoch_sizing(),
#         strategy='deepspeed_stage_3_offload',
#         precision=16,
#         callbacks=callbacks
#     )

#     trainer.strategy.load_full_weights = True

#     trainer.fit(model, data_loaders['train'], data_loaders['val'])

#     model = convert_zero_checkpoint_to_fp32_state_dict(f'{logger.base_dir}/lightning_logs/best_model.ckpt', f'{logger.base_dir}/best_model.ckpt')
    
#     trainer.test(model=model, ckpt_path=f'{logger.base_dir}/best_model.ckpt', dataloaders=data_loaders['test'])

#     loss_plot = data_explore.create_loss_plot(json_path=f'{logger.base_dir}/{logger.experiment_name}_loss.json')
#     logger.save_plot(loss_plot, "training_validation_loss")

#     if pred_model.task == 'classification':
#         fig, _ = data_explore.plot_roc_curve(json_path=f'{logger.base_dir}/{logger.experiment_name}_metrics.json')
#         logger.save_plot(fig, 'roc_curve')
#         fig = data_explore.plot_confusion_matrix_heatmap(json_path=f'{logger.base_dir}/{logger.experiment_name}_metrics.json')
#         logger.save_plot(fig, 'confusion_matrix')
#     elif pred_model.task == 'regression':
#         fig = data_explore.plot_actual_vs_predicted(json_path=f'{logger.base_dir}/{logger.experiment_name}_metrics.json')
#         logger.save_plot(fig, 'actual_vs_predicted')


def runner(config, embeddings, scores, logger, split=None, on_ray_tuning=False, num_workers=0, weights=None):
    head_config = config if not on_ray_tuning else utils.adjust_config_to_int(config)

    training_params = head_config['training_parameters']
    data_loaders = utils.create_data_loaders(
            embeddings, scores, scaler=training_params['scaler'], batch_size=training_params['batch_size'], validation_size=training_params['val_split'], split=split, num_workers=num_workers, weights=weights)
     
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

def ray_tuning(function_to_run, config, embeddings, scores, logger, experiment_dir, split=None, weights=None):

    network_type = config['architecture_parameters']['network_type']
    trials = 500 if network_type == 'mlp' else 100
    
    config['training_parameters']['learning_rate'] = (1e-6, 1e-2)
    config['training_parameters']['batch_size'] = (8, 128)
    config['training_parameters']['weight_decay'] = (1e-6, 1e-1)
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
        weights=weights,
        on_ray_tuning=True
    )
    
    best_config, best_loss = tuner.fit()

    logger.log(f"Best trial config: {best_config}")
    logger.log(f"Best trial metrics: {best_loss}")

    return best_config
