import torch
from plmfit.shared_utils import utils, data_explore
import plmfit.models.downstream_heads as heads
from plmfit.models.fine_tuning import LowRankAdaptationFineTuner
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from plmfit.models.lightning_model import LightningModel
from lightning.pytorch.strategies import DeepSpeedStrategy

def lora(args, logger):
    # Load dataset
    data = utils.load_dataset(args.data_type)
    if args.experimenting == "True": data = data.sample(1000)
    split = None if args.split is None else data[args.split]

    model = utils.init_plm(args.plm, logger)
    assert model != None, 'Model is not initialized'
    utils.disable_dropout(model)
    
    model.set_layer_to_use(args.layer)

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
    meta_data = None # This is only used for 'mut_mean' reduction
    if args.reduction == 'mut_mean': 
        meta_data = data['mut_mask'].values
    model.py_model.reduction = args.reduction
    
    encs = model.categorical_encode(data)

    scores = data['score'].values if head_config['architecture_parameters']['task'] == 'regression' else data['binary_score'].values
    training_params = head_config['training_parameters']
    data_loaders = utils.create_data_loaders(
            encs, scores, scaler=training_params['scaler'], 
            batch_size=training_params['batch_size'], 
            validation_size=training_params['val_split'], 
            dtype=torch.int8, 
            split=split,
            num_workers=0,
            meta_data=meta_data
        )
    
    fine_tuner = LowRankAdaptationFineTuner(training_config=training_params, logger=logger)
    model = fine_tuner.prepare_model(model, target_layers=args.target_layers)
 
    utils.trainable_parameters_summary(model, logger)
    model.py_model.task = pred_model.task
    
    model = LightningModel(model.py_model, head_config['training_parameters'], plmfit_logger=logger, log_interval=100)
    lightning_logger = TensorBoardLogger(save_dir=logger.base_dir, version=0, name="lightning_logs")
    model.experimenting = args.experimenting == "True" # If we are in experimenting mode

    strategy = DeepSpeedStrategy(
        stage=3,
        offload_optimizer=True,
        offload_parameters=True,
        load_full_weights = True,
        initial_scale_power = 20,
        loss_scale_window = 2000,
        min_loss_scale = 0.25
    )

    devices = args.gpus if torch.cuda.is_available() else 1
    strategy = strategy if torch.cuda.is_available() else 'auto'

    trainer = Trainer(
        default_root_dir=logger.base_dir,
        logger=lightning_logger, 
        max_epochs=model.hparams.epochs, 
        enable_progress_bar=False, 
        accumulate_grad_batches=model.gradient_accumulation_steps(),
        gradient_clip_val=model.gradient_clipping(),
        limit_train_batches=model.epoch_sizing(),
        limit_val_batches=model.epoch_sizing(),
        devices=devices,
        strategy=strategy,
        precision="16-mixed",
        callbacks=[model.early_stopping()]
    )

    if torch.cuda.is_available(): estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=int(args.gpus), num_nodes=1)

    try:
        trainer.fit(model, data_loaders['train'], data_loaders['val'])
    except Exception as e:
        # Check if the specific deepspeed minimum loss scale exception is raised
        if "Current loss scale already at minimum" in str(e):
            logger.log("Minimum loss reached. Ending training...")
        else:
            # If it's a different kind of exception, you might want to re-raise it
            raise e

    if torch.cuda.is_available(): model = convert_zero_checkpoint_to_fp32_state_dict(f'{logger.base_dir}/lightning_logs/best_model.ckpt', f'{logger.base_dir}/best_model.ckpt')
    
    trainer.test(model=model, ckpt_path=f'{logger.base_dir}/best_model.ckpt', dataloaders=data_loaders['test'])

    loss_plot = data_explore.create_loss_plot(json_path=f'{logger.base_dir}/{logger.experiment_name}_loss.json')
    logger.save_plot(loss_plot, "training_validation_loss")

    if pred_model.task == 'classification':
        fig, _ = data_explore.plot_roc_curve(json_path=f'{logger.base_dir}/{logger.experiment_name}_metrics.json')
        logger.save_plot(fig, 'roc_curve')
        fig = data_explore.plot_confusion_matrix_heatmap(json_path=f'{logger.base_dir}/{logger.experiment_name}_metrics.json')
        logger.save_plot(fig, 'confusion_matrix')
    elif pred_model.task == 'regression':
        fig = data_explore.plot_actual_vs_predicted(json_path=f'{logger.base_dir}/{logger.experiment_name}_metrics.json')
        logger.save_plot(fig, 'actual_vs_predicted')