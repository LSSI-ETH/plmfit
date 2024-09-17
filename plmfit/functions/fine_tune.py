import torch
from plmfit.shared_utils import utils, data_explore
import plmfit.models.downstream_heads as heads
from plmfit.models.fine_tuners import FullRetrainFineTuner, LowRankAdaptationFineTuner, BottleneckAdaptersFineTuner
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from plmfit.models.lightning_model import LightningModel
from lightning.pytorch.strategies import DeepSpeedStrategy
import ast
import shutil

def fine_tune(args, logger):
    head_config = utils.load_config(args.head_config)
    task = head_config['architecture_parameters']['task']

    # Load dataset
    data = utils.load_dataset(args.data_type)
    # data = data.sample(1000)
    # if args.experimenting == "True": data = data.sample(100)
    
    # This checks if args.split is set to 'sampled' and if 'sampled' is not in data, or if args.split is not a key in data.
    split = None if args.split == 'sampled' and 'sampled' not in data else data.get(args.split)
    model = utils.init_plm(args.plm, logger, task=task)
    assert model != None, 'Model is not initialized'

    if args.zeroed == "True":
        model.zeroed_model()

    model.experimenting = args.experimenting == "True" # If we are in experimenting mode
    
    model.set_layer_to_use(args.layer)

    logger.save_data(vars(args), 'arguments')
    logger.save_data(head_config, 'head_config')

    if task == 'masked_lm': data_loaders, training_params = masked_lm_prep(model=model, args=args, data=data, split=split, task=task, head_config=head_config, logger=logger)
    else: data_loaders, training_params = downstream_prep(model=model, args=args, data=data, split=split, task=task, head_config=head_config)
    
    if  args.ft_method == 'lora': fine_tuner = LowRankAdaptationFineTuner(training_config=training_params, logger=logger)
    elif args.ft_method == 'bottleneck_adapters': fine_tuner = BottleneckAdaptersFineTuner(training_config=training_params, logger=logger)
    elif args.ft_method == 'full': fine_tuner = FullRetrainFineTuner(training_config=training_params, logger=logger)
    else: raise ValueError('Fine Tuning method not supported')

    model = fine_tuner.prepare_model(model, target_layers=args.target_layers)
 
    utils.trainable_parameters_summary(model, logger)
    model.py_model.task = task
    
    model = LightningModel(model.py_model, training_params, plmfit_logger=logger, log_interval=100, experimenting=model.experimenting)
    lightning_logger = TensorBoardLogger(save_dir=logger.base_dir, version=0, name="lightning_logs")

    # TODO make this through the configuration defined
    if args.data_type == 'gb1' and args.split == 'one_vs_rest': model.track_validation_after = 10
    if args.data_type == 'rbd' and args.split == 'one_vs_rest': model.track_validation_after = -1
    if args.data_type == 'herH3' and args.split == 'one_vs_rest': model.track_validation_after = -1
    
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
        limit_train_batches=model.epoch_sizing() if len(data_loaders['train'].dataset) > 30000 else 1.0,
        limit_val_batches=model.epoch_sizing() if len(data_loaders['train'].dataset) > 30000 else 1.0,
        devices=devices,
        strategy=strategy,
        precision="16-mixed",
        callbacks=[model.early_stopping()]
    )
    if torch.cuda.is_available(): estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=int(args.gpus), num_nodes=1)

    model.train()
    trainer.fit(model, data_loaders['train'], data_loaders['val'])

    if torch.cuda.is_available(): 
        convert_zero_checkpoint_to_fp32_state_dict(f'{logger.base_dir}/lightning_logs/best_model.ckpt', f'{logger.base_dir}/best_model.ckpt')

    loss_plot = data_explore.create_loss_plot(json_path=f'{logger.base_dir}/{logger.experiment_name}_loss.json')
    logger.save_plot(loss_plot, "training_validation_loss")

    # TODO: Testing for lm 
    if task != 'masked_lm': trainer.test(model=model, ckpt_path=f'{logger.base_dir}/best_model.ckpt', dataloaders=data_loaders['test'])

    if task == 'classification':
        if head_config['architecture_parameters']['output_dim'] == 1:
            fig, _ = data_explore.plot_roc_curve(json_path=f'{logger.base_dir}/{logger.experiment_name}_metrics.json')
            logger.save_plot(fig, 'roc_curve')
        fig = data_explore.plot_confusion_matrix_heatmap(json_path=f'{logger.base_dir}/{logger.experiment_name}_metrics.json')
        logger.save_plot(fig, 'confusion_matrix')
    elif task == 'regression':
        fig = data_explore.plot_actual_vs_predicted(json_path=f'{logger.base_dir}/{logger.experiment_name}_metrics.json')
        logger.save_plot(fig, 'actual_vs_predicted')
    
    if torch.cuda.is_available():
        shutil.rmtree(f'{logger.base_dir}/lightning_logs/best_model.ckpt')
        shutil.rmtree(f'{logger.base_dir}/lightning_logs/version_0/checkpoints')

def downstream_prep(model, args, data, split, task, head_config):
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

    encs = model.categorical_encode(data)

    if task == 'regression':
        scores = data['score'].values 
    elif task == 'classification':
        if 'binary_score' in data:
            scores = data['binary_score'].values
        elif 'label' in data:
            scores = data['label'].values
        else:
            raise KeyError("Neither 'binary_score' nor 'label' found in data")
    else:
        raise ValueError('Task not supported')

    training_params = head_config['training_parameters']
    data_loaders = utils.create_data_loaders(
            encs, scores, scaler=training_params['scaler'], 
            batch_size=training_params['batch_size'], 
            validation_size=training_params['val_split'], 
            dtype=torch.int8, 
            split=split,
            num_workers=0
        )
    
    return data_loaders, training_params

def masked_lm_prep(model, args, data, split, task, head_config, logger):
    mut_mask = None # This is only used for 'mut_mean' reduction
    try:
        mut_mask = [ast.literal_eval(m) for m in data['mut_mask'].values]
        # Add 1 to each position to account for the added BOS token when the sequence will be encoded
        mut_mask = [[position + 1 for position in sublist] for sublist in mut_mask]
    except:
        mut_mask = None
        logger.log("No mutation information found. Reverting to full random masking...")     
    
    tokenizer = utils.load_transformer_tokenizer(args.plm, model.tokenizer)
    encoded_inputs = tokenizer(data['aa_seq'].to_list(), padding=True, return_tensors='pt', return_special_tokens_mask=True, add_special_tokens=True)
    
    # Initialize mutation_mask with zeros
    mutation_masks = []
    for idx, seq in enumerate(data['aa_seq']):
        seq_len = len(encoded_inputs['input_ids'][idx])
        mask = torch.zeros(seq_len, dtype=torch.int)
        if mut_mask and idx < len(mut_mask):
            # Set positions in the mask to 1 where mutations are present
            mutation_positions = mut_mask[idx]
            mask[mutation_positions] = 1
        mutation_masks.append(mask)
        
    # Convert list of masks to a tensor
    mutation_masks = torch.stack(mutation_masks)

    # Include the mutation_masks in the encoded_inputs
    encoded_inputs['mutation_mask'] = mutation_masks

    mutation_boost_factor = 1.0/head_config['architecture_parameters']['mlm_probability']
    training_params = head_config['training_parameters']
    train_size = 1.0 - training_params['val_split'] - training_params['test_split']
    val_size = training_params['val_split']
    test_size = training_params['test_split']
    
    # Create data loaders
    data_loaders = utils.create_mlm_data_loaders(
        encoded_inputs, 
        tokenizer, 
        batch_size=training_params['batch_size'], 
        mlm_probability=head_config['architecture_parameters']['mlm_probability'],
        mutation_boost_factor=mutation_boost_factor,
        split_ratios=(train_size, val_size, test_size)
    )

    return data_loaders, training_params