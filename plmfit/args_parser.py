import argparse

from attr import dataclass

def parse_args():
    """
    Parse arguments for the plmfit functions
    """
    parser = argparse.ArgumentParser(description='plmfit_args')
    args = DefaultArgs()
    parser.add_argument('--plm', type=str, default=args.plm)
    parser.add_argument('--ft_method', type=str, default=args.ft_method)
    parser.add_argument('--target_layers', type=str, default=args.target_layers)
    parser.add_argument('--data_type', type=str, default=args.data_type)
    parser.add_argument('--head_config', type=str, default=args.head_config)
    parser.add_argument('--ray_tuning', type=str, default=args.ray_tuning)
    parser.add_argument('--split', default=args.split)
    parser.add_argument('--function', type=str, default=args.function)
    parser.add_argument('--reduction', type=str, default=args.reduction)
    parser.add_argument('--layer', type=str, default=args.layer)
    parser.add_argument('--output_dir', type=str, default=args.output_dir)
    parser.add_argument('--experiment_name', type=str, default=args.experiment_name)
    parser.add_argument('--experiment_dir', type=str, default=args.experiment_dir)
    parser.add_argument('--embeddings_path', type=str, default=args.embeddings_path)
    parser.add_argument('--logger', type=str, default=args.logger)
    parser.add_argument('--cpus', default=args.cpus)
    parser.add_argument('--gpus', default=args.gpus)
    parser.add_argument('--nodes', type=int, default=args.nodes)
    parser.add_argument('--beta', default=args.beta)
    parser.add_argument('--experimenting', default=args.experimenting)
    parser.add_argument('--zeroed', default=args.zeroed)
    parser.add_argument('--garbage', default=args.garbage)
    parser.add_argument('--nulled', default=args.nulled)
    parser.add_argument('--weights', default=args.weights)
    parser.add_argument('--sampler', default=args.sampler)
    parser.add_argument('--split_size', default=args.split_size, type=int)
    parser.add_argument('--model_path', default=args.model_path, help="Path of the model in .ckpt format for evaluating it or continuing training from checkpoint")
    parser.add_argument('--model_metadata', default=args.model_metadata, help="Path of the model metadata to load the model")
    parser.add_argument('--evaluate', default=args.evaluate)
    parser.add_argument('--seed', default=args.seed, type=int)
    parser.add_argument('--hyperparam_config', default=args.hyperparam_config, type=str)
    parser.add_argument('--prediction_data', default=args.prediction_data, type=str)
    parser.add_argument('--batch_size', default=args.batch_size, type=int, help="Batch size mainly used for prediction")

    return parser.parse_args()

@dataclass
class DefaultArgs:
    plm: str = "progen2-small"
    ft_method: str = "feature_extraction"
    target_layers: str = "all"
    data_type: str = "aav"
    head_config: str = "linear_head_config.json"
    ray_tuning: str = "False"
    split: str = "sampled"
    function: str = "extract_embeddings"
    reduction: str = "mean"
    layer: str = "last"
    output_dir: str = "./output"
    experiment_name: str = "default"
    experiment_dir: str = "./output"
    embeddings_path: str = None
    logger: str = "local"
    cpus: int = 1
    gpus: int = 0
    nodes: int = 1
    beta: str = "False"
    experimenting: str = "False"
    zeroed: str = "False"
    garbage: str = "False"
    nulled: str = "False"
    weights: str = None
    sampler: str = "False"
    split_size: int = 0
    model_path: str = None
    model_metadata: str = None
    evaluate: str = "False"
    seed: int = 42
    hyperparam_config: str = "hyperparam_config.json"
    prediction_data: str = None
    batch_size: int = 100