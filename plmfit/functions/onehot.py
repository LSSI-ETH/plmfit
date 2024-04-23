from plmfit.shared_utils import utils
import torch
import torch.nn.functional as F
import plmfit.models.downstream_heads as heads
from plmfit.models.fine_tuning import FullRetrainFineTuner
import ray
from ray.tune import CLIReporter
from ray.train import RunConfig
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter

def onehot(args, logger):
    # Load dataset
    data = utils.load_dataset(args.data_type)
    split = None if args.split is None else data[args.split]
    head_config = utils.load_config(args.head_config)

    tokenizer = utils.load_tokenizer('proteinbert') # Use same tokenizer as proteinbert
    encs = utils.categorical_encode(
        data['aa_seq'].values, tokenizer, max(data['len'].values), add_eos=True, logger=logger, model_name='proteinbert')
    encs = F.one_hot(encs, tokenizer.get_vocab_size())
    encs = encs.reshape(encs.shape[0], -1)

    scores = data['score'].values if head_config['architecture_parameters']['task'] == 'regression' else data['binary_score'].values
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
    fine_tuner.train(pred_model, dataloaders_dict=data_loaders, on_ray_tuning=on_ray_tuning)

def ray_tuning(function_to_run, config, encodings, scores, logger, experiment_dir, split=None):
    network_type = config['architecture_parameters']['network_type']

    # Define search space and number of trials
    trials = 100
    if network_type == 'mlp': 
        config['architecture_parameters']['hidden_dim'] = ray.tune.uniform(64, 2048)
        trials = 200
    config['training_parameters']['learning_rate'] = ray.tune.uniform(1e-6, 1e-3)
    config['training_parameters']['batch_size'] = ray.tune.uniform(4, 256)
    config['training_parameters']['weight_decay'] = ray.tune.uniform(1e-4, 1e-2)

    initial_epoch_sizing = config['training_parameters']['epoch_sizing']
    config['training_parameters']['epoch_sizing'] = 0.2 # Sample data to make procedure faster

    # Initialize BayesOptSearch
    searcher = BayesOptSearch(
        utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0}, random_search_steps=10
    )
    # searcher = ConcurrencyLimiter(searcher, max_concurrent=4)

    scheduler = ASHAScheduler(
        max_t=10,
        grace_period=2,
        reduction_factor=2
    )

    reporter = CLIReporter(max_progress_rows=10)

    logger.log("Initializing ray tuning...")
    ray.init(include_dashboard=True)

    logger.mute = True # Avoid overpopulating logger with a mixture of training procedures
    tuner = ray.tune.Tuner(
        ray.tune.with_resources(
            ray.tune.with_parameters(
                function_to_run,
                encodings=encodings, 
                scores=scores,
                logger=logger,
                split=split, on_ray_tuning=True
            ), 
            {"gpu": 1}
        ),
        tune_config=ray.tune.TuneConfig(
            metric="loss", 
            mode="min",
            search_alg=searcher,
            scheduler=scheduler,
            num_samples=trials,
        ),
        run_config=RunConfig(
            progress_reporter=reporter, 
            log_to_file=(f"{experiment_dir}/ray_stdout.log", f"{experiment_dir}/ray_stderr.log"),
            storage_path=f'{experiment_dir}/raytune_results'),
        param_space=config,
    )
    results = tuner.fit()
    logger.mute = False # Ok, logger can be normal now

    best_result = results.get_best_result("loss", "min")
    best_result.config['training_parameters']['epoch_sizing'] = initial_epoch_sizing
    logger.log(f"Best trial config: {best_result.config}")
    logger.log(f"Best trial metrics: {best_result.metrics}")

    return best_result.config
