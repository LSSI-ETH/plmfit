import argparse

def parse_args():
    """
    Parse arguments for the plmfit functions
    """
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
    parser.add_argument('--model_path', default=None, help="Path of the model in .ckpt format for evaluating it or fine-tuning from checkpoint")
    parser.add_argument('--model_metadata', default=None, help="Path of the model metadata to load the model")
    parser.add_argument('--evaluate', default="False")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--hyperparam_config', default="hyperparam_config.json", type=str)
    parser.add_argument('--prediction_data', default=None, type=str)
    parser.add_argument('--batch_size', default=100, type=int, help="Batch size mainly used for prediction")
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint file to resume training from')

    return parser.parse_args()