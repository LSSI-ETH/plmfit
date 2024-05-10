from plmfit.logger import Logger
import os
import argparse
import traceback

NUM_WORKERS = 0

# TODO: Lightning implementation
def run_extract_embeddings(args, logger):
    from plmfit.functions import extract_embeddings
    extract_embeddings(args=args, logger=logger)

# TODO: Lightning implementation
def run_feature_extraction(args, logger):
    from plmfit.functions import feature_extraction
    feature_extraction(args=args, logger=logger)

def run_lora(args, logger):
    from plmfit.functions import lora
    lora(args=args, logger=logger)

def run_bottleneck_adapters(args, logger):
    from plmfit.functions import bottleneck_adapters
    bottleneck_adapters(args=args, logger=logger)

def run_full_retrain(args, logger):
    from plmfit.functions import full_retrain
    full_retrain(args=args, logger=logger)

# TODO: Lightning implementation
def run_onehot(args, logger):
    from plmfit.functions import onehot
    onehot(args, logger)

def run_developing(args, logger):
    from plmfit.functions import developing
    developing(args, logger)

def main():
    parser = argparse.ArgumentParser(description='plmfit_args')
    
    parser.add_argument('--plm', type=str, default='progen2-small')
    parser.add_argument('--ft_method', type=str, default='feature_extraction')
    parser.add_argument('--target_layers', type=str, default='all')
    parser.add_argument('--data_type', type=str, default='aav')
    parser.add_argument('--head_config', type=str, default='linear_head_config.json')
    parser.add_argument('--ray_tuning', type=str, default="False")
    parser.add_argument('--split', default=None)
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
    parser.add_argument('--cpus', default=1)
    parser.add_argument('--gpus', default=0)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--beta', default="False")
    parser.add_argument('--experimenting', default="False")

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
    try:
        if args.function == 'extract_embeddings': run_extract_embeddings(args, logger)
        elif args.function == 'fine_tuning':
            if args.ft_method == 'feature_extraction': run_feature_extraction(args, logger)
            elif args.ft_method == 'lora': run_lora(args, logger)
            elif args.ft_method == 'bottleneck_adapters': run_bottleneck_adapters(args, logger)
            elif args.ft_method == 'full': run_full_retrain(args, logger)
            else: raise ValueError('Fine Tuning method not supported')
        elif args.function == 'one_hot': run_onehot(args, logger)
        elif args.function == 'developing': run_developing(args, logger) # For developing new functions and testing them
        else: raise ValueError('Function not supported')
        logger.log("\n\nEnd of process", force_send=True)
    except:
        logger.mute = False
        stack_trace = traceback.format_exc()
        logger.log(stack_trace, force_send=True)

if __name__ == '__main__':
    main()