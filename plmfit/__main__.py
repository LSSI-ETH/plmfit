from plmfit.logger import Logger
import os
import argparse
import traceback
from plmfit.shared_utils.random_state import set_seed

NUM_WORKERS = 0

# TODO: Lightning implementation
def run_extract_embeddings(args, logger):
    from plmfit.functions import extract_embeddings
    extract_embeddings(args=args, logger=logger)

# TODO: Lightning implementation
def run_feature_extraction(args, logger):
    from plmfit.functions import feature_extraction
    feature_extraction(args=args, logger=logger)

def run_fine_tuning(args, logger):
    from plmfit.functions import fine_tune
    fine_tune(args=args, logger=logger)

# TODO: Lightning implementation
def run_onehot(args, logger):
    from plmfit.functions import onehot
    onehot(args, logger)

# TODO: Lightning implementation
def run_blosum(args, logger):
    from plmfit.functions import blosum
    blosum(args, logger)

def run_predict(args, logger):
    raise NotImplementedError("Function not supported (yet)")

def main():
    parser = argparse.ArgumentParser(description='plmfit_args')
    
    parser.add_argument('--plm', type=str, default='progen2-small')
    parser.add_argument('--ft_method', type=str, default='feature_extraction')
    parser.add_argument('--target_layers', type=str, default='all')
    parser.add_argument('--data_type', type=str, default='aav')
    parser.add_argument('--head_config', type=str, default='linear_head_config.json')
    parser.add_argument('--ray_tuning', type=str, default="False")
    parser.add_argument('--split', default='sampled')
    parser.add_argument('--function', type=str, default='extract_embeddings')
    parser.add_argument('--reduction', type=str, default='mean',
                        help='Reduction technique')
    parser.add_argument('--layer', type=str, default='last',
                        help='PLM layer to be used')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for created files')
    parser.add_argument('--experiment_name', type=str, default='default',
                        help='Output directory for created files')
    parser.add_argument('--experiment_dir', type=str, default='./output',
                        help='Output directory for created files (Keep the same as output_dir)')
    parser.add_argument('--embeddings_path', type=str, default=None,
                        help='Path where embeddings are stored')
    parser.add_argument('--logger', type=str, default='local')
    parser.add_argument('--cpus', default=1)
    parser.add_argument('--gpus', default=0)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--beta', default="False")
    parser.add_argument('--experimenting', default="False")
    parser.add_argument('--zeroed', default="False")
    parser.add_argument('--garbage', default="False")
    parser.add_argument('--nulled', default="False")
    parser.add_argument('--weights', default=None)
    parser.add_argument('--sampler', default="False")
    parser.add_argument('--split_size', default=0, type=int)
    parser.add_argument('--model_path', default=None, help="Path of the model in .ckpt format for evaluating it or continuing training from checkpoint")
    parser.add_argument('--evaluate', default="False")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--hyperparam_config', default="hyperparam_config.json", type=str)

    args = parser.parse_args()
    experiment_dir = args.experiment_dir
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir, exist_ok=True)

    # Set global seed
    set_seed(args.seed)
        
    # Removing the output_dir prefix from experiment_dir
    trimmed_experiment_dir = experiment_dir.removeprefix(f"{args.output_dir}/")
    logger = Logger(
        experiment_name = args.experiment_name, 
        base_dir= args.experiment_dir, 
        log_to_server=args.logger!='local', 
        server_path=f'{trimmed_experiment_dir}'
    )
    try:
        if args.function == 'extract_embeddings': run_extract_embeddings(args, logger)
        elif args.function == 'fine_tuning':
            if args.ft_method == 'feature_extraction': run_feature_extraction(args, logger) # TODO: Add this to fine tuning function as well
            else: run_fine_tuning(args, logger)
        elif args.function == 'one_hot': run_onehot(args, logger)
        elif args.function == 'blosum': run_blosum(args, logger)
        elif args.function == 'predict' or args.function == 'generate': run_predict(args, logger)
        else: raise NotImplementedError('Function not supported (yet)')
        logger.log("\n\nEnd of process", force_send=True)
    except:
        logger.mute = False
        stack_trace = traceback.format_exc()
        logger.log(stack_trace, force_send=True)

if __name__ == '__main__':
    main()