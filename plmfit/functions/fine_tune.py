import torch
from plmfit.shared_utils import utils, data_explore
import plmfit.models.downstream_heads as heads
from plmfit.models.fine_tuners import (
    FullRetrainFineTuner,
    LowRankAdaptationFineTuner,
    BottleneckAdaptersFineTuner,
)
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from plmfit.models.lightning_model import LightningModel
from lightning.pytorch.strategies import DeepSpeedStrategy
import ast
import shutil
from pathlib import Path
import numpy as np


def fine_tune(args, logger):
    head_config = utils.load_config(f"training/{args.head_config}")
    task = head_config["architecture_parameters"]["task"]
    print(f"Loaded head config: {head_config}") 
    print(f"Task: {task}") 

    data, split, weights, sampler = utils.data_pipeline(
        args.data_type,
        args.split,
        head_config["training_parameters"].get("weights", None),
        head_config["training_parameters"].get("sampler", False),
        dev=args.experimenting == "True",
    )

    print("\n=== DEBUG: Output from data_pipeline ===")
    print(f"Full dataset shape: {data.shape}")
    print(f"Split type: {type(split)}")
    if split is not None:
        print(f"Split sample (first 10): {split[:10]}")
        print(f"Unique split values: {set(split)}")
    else:
        print("Split is None!")

    print(f"Weights type: {type(weights)}")
    if weights is not None:
        print(f"Weights sample (first 10): {weights[:10]}")
    else:
        print("Weights is None!")

    print(f"Sampler: {sampler}")
    print("=== END DEBUG ===\n")

    print(f"Data loaded: {data.shape if isinstance(data, np.ndarray) else 'N/A'}")  # Check if data is loaded
    print(f"Split: {split}, Weights: {weights}, Sampler: {sampler}") 

    if args.evaluate == "True" and split is None:
        raise ValueError("Cannot evaluate without a standard testing split")

    model = utils.init_plm(args.plm, logger, task=task)
    print(f"Model initialized: {model}")

    if args.zeroed == "True":
        model.zeroed_model()
        print("Model has been zeroed.")

    model.experimenting = (
        args.experimenting == "True"
    )  # If we are in experimenting mode

    model.set_layer_to_use(args.layer)

    logger.save_data(vars(args), "arguments")
    logger.save_data(head_config, "head_config")

    if task == "masked_lm":
        data_loaders, training_params = masked_lm_prep(
            model=model,
            args=args,
            data=data,
            split=split,
            task=task,
            head_config=head_config,
            logger=logger,
        )
    else:
        data_loaders, training_params = downstream_prep(
            model=model,
            args=args,
            data=data,
            split=split,
            task=task,
            head_config=head_config,
            weights=weights,
            sampler=sampler,
        )
        print("Data loaders and training params for downstream prepared.")

    if args.ft_method == "lora":
        fine_tuner = LowRankAdaptationFineTuner(
            logger=logger
        )
        print("Using LowRankAdaptationFineTuner for fine-tuning.")
    elif args.ft_method == "bottleneck_adapters":
        fine_tuner = BottleneckAdaptersFineTuner(
            logger=logger
        )
    elif args.ft_method == "full":
        fine_tuner = FullRetrainFineTuner(
            logger=logger
        )
    else:
        raise ValueError("Fine-tuning method not supported")

    model = fine_tuner.prepare_model(model, target_layers=args.target_layers)
    print(f"Model prepared for fine-tuning with layers: {args.target_layers}")

    utils.trainable_parameters_summary(model, logger)
    
    model = LightningModel(
        model.py_model,
        training_params,
        plmfit_logger=logger,
        log_interval=100,
        experimenting=model.experimenting,
    )
    # if args.checkpoint is not None:
    #     # If ckpt_path is a zero checkpoint (check if it is a folder), convert it to fp32
    #     checkpoint = args.checkpoint
    #     if Path(args.checkpoint).is_dir():
    #         convert_zero_checkpoint_to_fp32_state_dict(
    #             args.checkpoint,
    #             f"{logger.base_dir}/checkpoint.ckpt",
    #         )
    #         checkpoint = f"{logger.base_dir}/checkpoint.ckpt"
    #     model = LightningModel.load_from_checkpoint(
    #         checkpoint, 
    #         model=model.model, 
    #         plmfit_logger=logger, 
    #         log_interval=100,
    #         experimenting=model.experimenting,
    #     )
    lightning_logger = TensorBoardLogger(
        save_dir=logger.base_dir, name="lightning_logs"
    )

    # TODO make this through the configuration defined
    if args.data_type == "gb1" and args.split == "one_vs_rest":
        model.track_validation_after = 10
    if args.data_type == "rbd" and args.split == "one_vs_rest":
        model.track_validation_after = -1
    if args.data_type == "herH3" and args.split == "one_vs_rest":
        model.track_validation_after = -1

    strategy = DeepSpeedStrategy(
        stage=3,
        offload_optimizer=True,
        offload_parameters=True,
        load_full_weights=False,
        initial_scale_power=20,
        loss_scale_window=2000,
        min_loss_scale=0.25,
        contiguous_gradients=True,
    )
    devices = args.gpus if torch.cuda.is_available() else 1
    strategy = strategy if torch.cuda.is_available() else "auto"

    print(f"Trainer setup:")
    print(f"Max epochs: {model.hparams.epochs}")
    print(f"Batch size: {model.hparams.batch_size}")
    print(f"Gradient accumulation: {model.gradient_accumulation_steps()}")
    print(f"Gradient clipping: {model.gradient_clipping()}")
    print(f"Limit train batches: {model.epoch_sizing()}")
    print(f"Devices: {devices}")
    print(f"Precision: {'16-mixed' if torch.cuda.is_available() else 32}")


    trainer = Trainer(
        default_root_dir=logger.base_dir,
        logger=lightning_logger,
        max_epochs=model.hparams.epochs,
        enable_progress_bar=False,
        accumulate_grad_batches=model.gradient_accumulation_steps(),
        gradient_clip_val=model.gradient_clipping(),
        limit_train_batches=(model.epoch_sizing()),
        limit_val_batches=(model.epoch_sizing()),
        devices=devices,
        strategy=strategy,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=[model.early_stopping()],
    )
    if torch.cuda.is_available():
        estimate_zero3_model_states_mem_needs_all_live(
            model, num_gpus_per_node=int(args.gpus), num_nodes=1
        )

    if args.evaluate != "True":
        print("Starting training...")
        model.train()
        trainer.fit(model, data_loaders["train"], data_loaders["val"], ckpt_path=args.checkpoint)

        ckpt_path = f"{logger.base_dir}/lightning_logs/best_model.ckpt"
        if torch.cuda.is_available():
            convert_zero_checkpoint_to_fp32_state_dict(
                f"{logger.base_dir}/lightning_logs/best_model.ckpt",
                f"{logger.base_dir}/best_model.ckpt",
            )
            ckpt_path = f"{logger.base_dir}/best_model.ckpt"

            loss_plot = data_explore.create_loss_plot(
                json_path=f"{logger.base_dir}/{logger.experiment_name}_loss.json"
            )
            logger.save_plot(loss_plot, "training_validation_loss")
    else:
        ckpt_path = args.model_path
        # If ckpt_path is a zero checkpoint (check if it is a folder), convert it to fp32
        if Path(ckpt_path).is_dir() and torch.cuda.is_available():
            convert_zero_checkpoint_to_fp32_state_dict(
                ckpt_path,
                f"{logger.base_dir}/best_model.ckpt",
            )

    # TODO: Testing for lm
    if task != "masked_lm":
        trainer.test(
            model=model,
            ckpt_path=ckpt_path,
            dataloaders=data_loaders["test"],
        )
    if task == "classification":
        if head_config["architecture_parameters"]["output_dim"] == 1:
            fig, _ = data_explore.plot_roc_curve(
                json_path=f"{logger.base_dir}/{logger.experiment_name}_metrics.json"
            )
            logger.save_plot(fig, "roc_curve")
        fig = data_explore.plot_confusion_matrix_heatmap(
            json_path=f"{logger.base_dir}/{logger.experiment_name}_metrics.json"
        )
        logger.save_plot(fig, "confusion_matrix")
    
    elif task == "multilabel_classification":
    # For multilabel classification, we handle each label as a separate binary classification task.
    # We will calculate the metrics for each label and plot accordingly.

        print("Plotting metrics for multilabel classification...")

        # If you want to plot multiple ROC curves, one for each label:
        # You could use a plotting function that handles multilabel ROC curves or use one of the following methods.
        # Example: plot ROC curves for each label
        fig, _ = data_explore.plot_multilabel_roc_curve(
            json_path=f"{logger.base_dir}/{logger.experiment_name}_metrics.json"
        )
        logger.save_plot(fig, "multilabel_roc_curve")

        # For the multilabel confusion matrix, each label has its own confusion matrix
        fig = data_explore.plot_multilabel_confusion_matrix_heatmap(
            json_path=f"{logger.base_dir}/{logger.experiment_name}_metrics.json"
        )
        logger.save_plot(fig, "multilabel_confusion_matrix")

    elif task == "regression":
        fig = data_explore.plot_actual_vs_predicted(
            json_path=f"{logger.base_dir}/{logger.experiment_name}_metrics.json"
        )
        logger.save_plot(fig, "actual_vs_predicted")
    elif task == "token_classification":
        fig = data_explore.plot_confusion_matrix_heatmap(
            json_path=f"{logger.base_dir}/{logger.experiment_name}_metrics.json"
        )
        logger.save_plot(fig, "confusion_matrix")

    if torch.cuda.is_available():
        best_model_file = f"{logger.base_dir}/lightning_logs/best_model.ckpt"
        if os.path.exists(best_model_file):
            os.remove(best_model_file)
        shutil.rmtree(f"{logger.base_dir}/lightning_logs/version_0/checkpoints")


def downstream_prep(
    model,
    args,
    data,
    split,
    task,
    head_config,
    weights=None,
    sampler=False,
):
    pred_model = heads.init_head(config=head_config, input_dim=model.emb_layers_dim)

    model.py_model.set_head(pred_model)
    model.py_model.reduction = args.reduction

    encs = model.categorical_encode(data)

    if task == "regression":
        scores = data["score"].values
    elif task == "classification":
        if "binary_score" in data:
            scores = data["binary_score"].values
        elif "label" in data:
            scores = data["label"].values
        else:
            raise KeyError("Neither 'binary_score' nor 'label' found in data")

    elif task == "multilabel_classification":
        # Labels are all columns starting with 'label_'
        scores = data[[col for col in data.columns if "label_" in col]].values
        # Ensure that -1 is replaced with -100
        scores[scores == -1] = -100
        
    elif task == "token_classification":
        scores = data["label"].values
        # Convert list of strings to list of list of integers
        scores = utils.convert_string_list_to_list_of_int_lists(scores)
        # Pad with -100 to match the sequence length
        scores = utils.pad_list_of_lists(
            scores,
            max(data["len"].values),
            pad_value=-100,
            convert_to_np=True,
            prepend_single_pad=True,
            append_single_pad=True,
        )
    else:
        raise ValueError("Task not supported")

    training_params = head_config["training_parameters"]
    data_loaders = utils.create_data_loaders(
        encs,
        scores,
        scaler=training_params["scaler"],
        batch_size=training_params["batch_size"],
        validation_size=training_params["val_split"],
        dtype=torch.int8,
        split=split,
        num_workers=0,
        weights=weights,
        sampler=sampler,
    )

    return data_loaders, training_params


def masked_lm_prep(model, args, data, split, task, head_config, logger):
    mut_mask = None  # This is only used for 'mut_mean' reduction
    try:
        mut_mask = [ast.literal_eval(m) for m in data["mut_mask"].values]
        # Add 1 to each position to account for the added BOS token when the sequence will be encoded
        mut_mask = [[position + 1 for position in sublist] for sublist in mut_mask]
    except:
        mut_mask = None
        logger.log("No mutation information found. Reverting to full random masking...")

    tokenizer = utils.load_transformer_tokenizer(args.plm, model.tokenizer)
    encoded_inputs = tokenizer(
        data["aa_seq"].to_list(),
        padding=True,
        return_tensors="pt",
        return_special_tokens_mask=True,
        add_special_tokens=True,
    )

    # Initialize mutation_mask with zeros
    mutation_masks = []
    for idx, seq in enumerate(data["aa_seq"]):
        seq_len = len(encoded_inputs["input_ids"][idx])
        mask = torch.zeros(seq_len, dtype=torch.int)
        if mut_mask and idx < len(mut_mask):
            # Set positions in the mask to 1 where mutations are present
            mutation_positions = mut_mask[idx]
            mask[mutation_positions] = 1
        mutation_masks.append(mask)

    # Convert list of masks to a tensor
    mutation_masks = torch.stack(mutation_masks)

    # Include the mutation_masks in the encoded_inputs
    encoded_inputs["mutation_mask"] = mutation_masks

    mutation_boost_factor = (
        1.0 / head_config["architecture_parameters"]["mlm_probability"]
    )
    training_params = head_config["training_parameters"]
    train_size = 1.0 - training_params["val_split"] - training_params["test_split"]
    val_size = training_params["val_split"]
    test_size = training_params["test_split"]

    # Create data loaders
    data_loaders = utils.create_mlm_data_loaders(
        encoded_inputs,
        tokenizer,
        batch_size=training_params["batch_size"],
        mlm_probability=head_config["architecture_parameters"]["mlm_probability"],
        mutation_boost_factor=mutation_boost_factor,
        split_ratios=(train_size, val_size, test_size),
    )

    return data_loaders, training_params
