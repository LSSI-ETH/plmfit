import json
import os

# === INPUTS ===
pretrained_config_DS = "/cluster/scratch/askrika/DS/one_hot/RBDDS_split_mlp_classification/20250423_132523_17128/RBDDS_split_mlp_classification_data.json"
pretrained_config_HC = "/cluster/scratch/askrika/HC/one_hot/RBDHC_split_mlp_classification/20250423_132519_27114/RBDHC_split_mlp_classification_data.json"

# Where to save processed head configs
output_base_dir_DS = "/cluster/home/askrika/ML/plmfit/config/training"
output_base_dir_HC = "/cluster/home/askrika/ML/plmfit/config/training"
#os.makedirs(output_base_dir, exist_ok=True)

def extract_head_config(full_config_path):
    with open(full_config_path, "r") as f:
        print(f"[INFO] Reading {full_config_path}")
        full_config = json.load(f)

    if "head_config" not in full_config:
        raise ValueError(f"No head_config in {full_config_path}")

    head_config = full_config["head_config"]
    arguments = full_config.get("arguments", {})

    return head_config, arguments

def make_filename(args, head_config):
    data_type = args.get("data_type", "unknown")
    split = args.get("split", "nosplit")
    task = args.get("task") or head_config["architecture_parameters"].get("task", "notask")
    return f"{data_type}_{split}_{task}_head_config.json"

# Extract + save for DS model
head_config_ds, args_ds = extract_head_config(pretrained_config_DS)
filename_ds = make_filename(args_ds, head_config_ds)
path_ds = os.path.join(output_base_dir_DS, filename_ds)

with open(path_ds, "w") as f:
    json.dump(head_config_ds, f, indent=2)
    print(f"[✓] Saved DS head_config to: {path_ds}")


# Extract + save for HC model
head_config_hc, args_hc = extract_head_config(pretrained_config_HC)
filename_hc = make_filename(args_hc, head_config_hc)
path_hc = os.path.join(output_base_dir_HC, filename_hc)

with open(path_hc, "w") as f:
    json.dump(head_config_hc, f, indent=2)
    print(f"[✓] Saved HC head_config to: {path_hc}")


'''
def run_evaluation_from_full_config(full_config_json_path, model_ckpt_path, new_data_type, new_output_dir):
    # Load JSON containing args + head_config
    with open(full_config_json_path, "r") as f:
        print("[DEBUG] Reading JSON from:", full_config_json_path)
        full_config = json.load(f)

    arguments = full_config["arguments"]
    head_config = full_config["head_config"]
    print("[DEBUG] Loaded hidden_dim from JSON:", head_config["architecture_parameters"]["hidden_dim"])

    os.makedirs(new_output_dir, exist_ok=True)
    config_path = os.path.join(new_output_dir, "cross_eval_head_config.json")
    with open(config_path, "w") as f:
        json.dump(head_config, f, indent=2)

    # Create symlink in config/training/
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    training_dir = os.path.join(project_root, "config", "training")
    os.makedirs(training_dir, exist_ok=True)
    symlink_target = os.path.join(training_dir, "cross_eval_head_config.json")
    if os.path.islink(symlink_target) or os.path.exists(symlink_target):
        os.remove(symlink_target)
    os.symlink(config_path, symlink_target)

    class Args:
        def __init__(self):
            self.function = arguments.get("function", "onehot")
            self.data_type = new_data_type
            self.split = arguments["split"]
            self.head_config = "cross_eval_head_config.json"
            self.ray_tuning = "False"
            self.evaluate = "True"
            self.model_path = model_ckpt_path
            self.output_dir = new_output_dir
            self.gpus = arguments.get("gpus", "1")
            self.experimenting = "False"
            self.experiment_name = f"{new_data_type}_cross_eval"
            self.experiment_dir = os.path.join(new_output_dir, self.experiment_name)
            self.base_dir = self.experiment_dir


    args = Args()
    print(f"[DEBUG] args.experiment_name is: {args.experiment_name}")
    print(f"[DEBUG] args.experiment_dir is: {args.experiment_dir}")
    logger = Logger(args)
    os.makedirs(logger.base_dir, exist_ok=True)
    print(f"[DEBUG] Logger base_dir is: {logger.base_dir}")

    onehot(args, logger)


def print_mcc_from_metrics(output_dir, experiment_name):
    metrics_path = os.path.join(output_dir, experiment_name, f"{experiment_name}_metrics.json")
    print("[DEBUG] Checking metrics at:", metrics_path)

    if not os.path.exists(metrics_path):
        print(f"[❌] Metrics file not found: {metrics_path}")
        return

    with open(metrics_path, "r") as f:
        data = json.load(f)
        mcc = data.get("mcc", None)
        if mcc is not None:
            print(f"[✅] MCC for {experiment_name}: {mcc:.4f}")
        else:
            print(f"[⚠️] MCC not found in {metrics_path}")


# Example usage
if __name__ == "__main__":
    hc_summary_json = "/cluster/scratch/askrika/HC/one_hot/RBDHC_split_mlp_classification/20250423_132519_27114/RBDHC_split_mlp_classification_data.json"
    ds_summary_json = "/cluster/scratch/askrika/DS/one_hot/RBDDS_split_mlp_classification/20250423_132523_17128/RBDDS_split_mlp_classification_data.json"
    
    hc_ckpt = "/cluster/scratch/askrika/HC/one_hot/RBDHC_split_mlp_classification/20250423_132519_27114/lightning_logs/best_model.ckpt"
    ds_ckpt = "/cluster/scratch/askrika/DS/one_hot/RBDDS_split_mlp_classification/20250423_132523_17128/lightning_logs/best_model.ckpt"

    hc_on_ds_out = "/cluster/scratch/askrika/cross_eval/hc_on_ds"
    ds_on_hc_out = "/cluster/scratch/askrika/cross_eval/ds_on_hc"

    # HC model on DS data
    run_evaluation_from_full_config(hc_summary_json, hc_ckpt, "RBDDS", hc_on_ds_out)
    print_mcc_from_metrics(hc_on_ds_out, "RBDDS_cross_eval")

    # DS model on HC data
    run_evaluation_from_full_config(ds_summary_json, ds_ckpt, "RBDHC", ds_on_hc_out)
    print_mcc_from_metrics(ds_on_hc_out, "RBDHC_cross_eval")
'''