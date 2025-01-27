from plmfit.logger import Logger
import os
from plmfit.args_parser import parse_args
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

def run_onehot(args, logger):
    from plmfit.functions import onehot
    onehot(args, logger)

def run_blosum(args, logger):
    from plmfit.functions import blosum
    blosum(args, logger)

def run_predict(args, logger):
    from plmfit.functions import predict
    predict(args, logger)

def main():
    args = parse_args()
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
        elif args.function == 'fine_tuning': run_fine_tuning(args, logger)
        elif args.function == 'feature_extraction': run_feature_extraction(args, logger)
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
