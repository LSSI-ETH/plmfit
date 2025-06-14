import torch
import plmfit.models.downstream_heads as heads
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from plmfit.models.lightning_model import LightningModel
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import lightning.pytorch as pl
from optuna.integration import PyTorchLightningPruningCallback
from packaging import version
from optuna.visualization import plot_optimization_history, plot_slice
from plmfit.shared_utils import utils, data_explore
from plmfit.logger import LogOptunaTrialCallback
import gc
import copy
import pandas as pd
from typing import Optional


def feature_extraction(args, logger, data: Optional[pd.DataFrame] = None, head_config: Optional[dict] = None):
    head_config = utils.load_config(f"training/{args.head_config}") if not isinstance(head_config, dict) else head_config
    task = head_config["architecture_parameters"]["task"]

    data, split, weights, sampler = utils.data_pipeline(
        args.data_type if data is None else data,
        args.split,
        head_config["training_parameters"].get("weights", None),
        head_config["training_parameters"].get("sampler", False),
        dev=args.experimenting == "True",
    )

    if args.evaluate == "True" and split is None:
        raise ValueError("Cannot evaluate without a standard testing split")

    ### TODO: Extract embeddings if do not exist & get path optionally from args
    embeddings = utils.load_embeddings(
        emb_path=(
            f"{args.output_dir}/extract_embeddings"
            if args.embeddings_path is None
            else args.embeddings_path
        ),
        data_type=args.data_type,
        model=args.plm,
        layer=args.layer,
        reduction=args.reduction,
    )
    assert (
        embeddings != None
    ), "Couldn't find embeddings, use the full path of the embeddings file (.pt) or you can use extract_embeddings function to create and save the embeddings."

    if task == "regression":
        scores = data["score"].values
    elif task == "classification":
        if "binary_score" in data:
            scores = data["binary_score"].values
        elif "label" in data:
            scores = data["label"].values
        else:
            raise KeyError("Neither 'binary_score' nor 'label' found in data")
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
    elif task == "multilabel_classification":
        # Labels are all columns starting with 'label_'
        scores = data[[col for col in data.columns if "label_" in col]].values

        # Replace -1 with -100
        scores[scores == -1] = -100
    else:
        raise ValueError("Task not supported")

    if args.ray_tuning == "True":
        assert args.evaluate == "False", "Cannot evaluate and tune at the same time"

        head_config = hyperparameter_tuning(
            task,
            args,
            head_config,
            embeddings,
            scores,
            logger,
            split=split,
            num_workers=0,
            weights=weights,
            sampler=sampler,
        )

    logger.save_data(vars(args), "arguments")
    logger.save_data(head_config, "head_config")

    objective(
        None,
        task,
        args,
        head_config,
        embeddings,
        scores,
        logger,
        split=split,
        on_ray_tuning=False,
        num_workers=0,
        weights=weights,
        sampler=sampler,
    )


def objective(
    trial,
    task,
    args,
    head_config,
    embeddings,
    scores,
    logger,
    split=None,
    on_ray_tuning=False,
    num_workers=0,
    weights=None,
    sampler=False,
    patience=5,
    hyperparam_config=None,
):
    config = copy.deepcopy(head_config)

    network_type = config["architecture_parameters"]["network_type"]

    epochs = config["training_parameters"]["epochs"]
    if trial is not None and on_ray_tuning:
        epochs = int(config["training_parameters"]["epochs"] // (1 / hyperparam_config["epochs_fragment"]))
        print(epochs)
        for param_name, param_info in hyperparam_config["architecture_parameters"].items():
            p_type = param_info["type"]                # "float" or "int"
            p_range = param_info["range"] 
            config["architecture_parameters"][param_name] = utils.suggest_number_of_type(trial, param_name, p_range[0],
                    p_range[1], p_type)
        for param_name, param_info in hyperparam_config["training_parameters"].items():
            p_type = param_info["type"]                # "float" or "int"
            p_range = param_info["range"] 
            config["training_parameters"][param_name] = utils.suggest_number_of_type(trial, param_name, p_range[0],
                    p_range[1], p_type)

    training_params = config["training_parameters"]
    data_loaders = utils.create_data_loaders(
        embeddings,
        scores,
        scaler=training_params["scaler"],
        batch_size=training_params["batch_size"],
        validation_size=training_params["val_split"],
        split=split,
        num_workers=num_workers,
        weights=weights,
        sampler=sampler,  
    )

    model = heads.init_head(
        config=config, input_dim=embeddings.shape[-1]
    )

    if not on_ray_tuning:
        logger.save_data(config, "head_config")

    utils.set_trainable_parameters(model)

    model = LightningModel(
        model,
        training_params,
        plmfit_logger=logger,
        log_interval=100 if not on_ray_tuning else -1,
    )
    lightning_logger = TensorBoardLogger(
        save_dir=logger.base_dir, version=0, name="lightning_logs"
    )

    # TODO make this through the configuration defined
    if args.data_type == "gb1" and args.split == "one_vs_rest":
        model.track_validation_after = 10
    if args.data_type == "rbd" and args.split == "one_vs_rest":
        model.track_validation_after = -1
    if args.data_type == "herH3" and args.split == "one_vs_rest":
        model.track_validation_after = -1

    devices = 1
    strategy = "auto"

    callbacks = []
    epoch_sizing = model.epoch_sizing()
    if on_ray_tuning:
        epoch_sizing = hyperparam_config["epoch_sizing"]
        callbacks.append(PyTorchLightningPruningCallback(trial, monitor=f"val_loss"))
    callbacks.append(model.early_stopping() if not on_ray_tuning else model.early_stopping(patience))

    trainer = Trainer(
        default_root_dir=logger.base_dir,
        logger=lightning_logger,
        enable_checkpointing=False,
        max_epochs=epochs,
        enable_progress_bar=False,
        accumulate_grad_batches=model.gradient_accumulation_steps(),
        gradient_clip_val=model.gradient_clipping(),
        limit_train_batches=epoch_sizing,
        limit_val_batches=epoch_sizing,
        devices=devices,
        strategy=strategy,
        precision="16-mixed",
        callbacks=callbacks,
    )

    if on_ray_tuning:
        hyperparameters = dict()
        for param_name, _ in hyperparam_config["architecture_parameters"].items():
            hyperparameters[param_name] = config["architecture_parameters"][param_name]
        for param_name, _ in hyperparam_config["training_parameters"].items():
            hyperparameters[param_name] = config["training_parameters"][param_name]

        trainer.logger.log_hyperparams(hyperparameters)

    if args.evaluate != "True":
        model.train()
        trainer.fit(model, data_loaders["train"], data_loaders["val"])

        if on_ray_tuning:
            callbacks[0].check_pruned()
            return trainer.callback_metrics[f"val_loss"].item()
        
        loss_plot = data_explore.create_loss_plot(
            json_path=f"{logger.base_dir}/{logger.experiment_name}_loss.json"
        )
        logger.save_plot(loss_plot, "training_validation_loss")
        ckpt_path = f"{logger.base_dir}/lightning_logs/best_model.ckpt"
    else:
        ckpt_path = args.model_path
    trainer.test(
        model=model,
        ckpt_path=ckpt_path,
        dataloaders=data_loaders["test"],
    )

    if task == "classification":
        if config["architecture_parameters"]["output_dim"] == 1:
            fig, _ = data_explore.plot_roc_curve(
                json_path=f"{logger.base_dir}/{logger.experiment_name}_metrics.json"
            )
            logger.save_plot(fig, "roc_curve")
        fig = data_explore.plot_confusion_matrix_heatmap(
            json_path=f"{logger.base_dir}/{logger.experiment_name}_metrics.json"
        )
        logger.save_plot(fig, "confusion_matrix")
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


def hyperparameter_tuning(
    task,
    args,
    head_config,
    embeddings,
    scores,
    logger,
    split=None,
    num_workers=0,
    weights=None,
    sampler=False,
):
    if version.parse(pl.__version__) < version.parse("2.2.1"):
        raise RuntimeError(
                "PyTorch Lightning>=2.2.1 is required for hyper-parameter tuning."
            )
    network_type = head_config["architecture_parameters"]["network_type"]
    hyperparam_config = utils.load_config(f"training/hyperparams/{args.hyperparam_config}")
    network_config = hyperparam_config[network_type]

    # 1. Number of trials
    n_trials = network_config["n_trials"]

    storage = JournalStorage(
        JournalFileBackend(f"{logger.base_dir}/optuna_journal_storage.log")
    )
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        study_name="plmfit",
        storage=storage,
        load_if_exists=True,
    )

    logger.log("Starting hyperparameter tuning...")
    logger.mute = True
    study.optimize(
        lambda trial: objective(
            trial,
            task,
            args,
            head_config,
            embeddings,
            scores,
            logger,
            split,
            on_ray_tuning=True,
            num_workers=num_workers,
            weights=weights,
            sampler=sampler,
            hyperparam_config=network_config,
        ),
        n_trials=n_trials,
        callbacks=[LogOptunaTrialCallback(logger)],
        n_jobs=int(args.gpus),
        gc_after_trial=True,
        catch=(FileNotFoundError,),
    )
    logger.mute = False
    history = plot_optimization_history(study)
    slice = plot_slice(study)
    history.write_image(f"{logger.base_dir}/optuna_optimization_history.png")
    slice.write_image(f"{logger.base_dir}/optuna_slice.png")

    for param_name, _ in network_config["architecture_parameters"].items():
        head_config["architecture_parameters"][param_name] = study.best_params[param_name]
    for param_name, _ in network_config["training_parameters"].items():
        head_config["training_parameters"][param_name] = study.best_params[param_name]

    return head_config
