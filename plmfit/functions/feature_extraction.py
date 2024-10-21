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


def feature_extraction(args, logger):
    head_config = utils.load_config(f"training/{args.head_config}")
    task = head_config["architecture_parameters"]["task"]

    # Load dataset
    data = utils.load_dataset(args.data_type)
    # data = data.sample(30000)
    # if args.experimenting == "True": data = data.sample(100)

    # This checks if args.split is set to 'sampled' and if 'sampled' is not in data, or if args.split is not a key in data.
    split = (
        None
        if args.split == "sampled" and "sampled" not in data
        else data.get(args.split)
    )

    # Load class weights if they exist
    weights = (
        None
        if head_config["training_parameters"].get("weights") is None
        else data.get(head_config["training_parameters"]["weights"])
    )
    sampler = head_config["training_parameters"].get("sampler", False) == True

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

    if args.ray_tuning == "True":
        head_config = hyperparameter_tuning(
            task,
            args,
            head_config,
            embeddings,
            data,
            logger,
            split=split,
            num_workers=0,
            weights=weights,
            sampler=sampler,
            n_trials=100,
        )

    logger.save_data(vars(args), "arguments")
    logger.save_data(head_config, "head_config")

    objective(
        None,
        task,
        args,
        head_config,
        embeddings,
        data,
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
    data,
    logger,
    split=None,
    on_ray_tuning=False,
    num_workers=0,
    weights=None,
    sampler=False,
    patience=5,
):
    config = copy.deepcopy(head_config)

    network_type = config["architecture_parameters"]["network_type"]

    epochs = config["training_parameters"]["epochs"]
    if trial is not None and on_ray_tuning:
        epochs = config["training_parameters"]["epochs"] // 4
        config["training_parameters"]["learning_rate"] = trial.suggest_float(
            "learning_rate", 1e-6, 1e-2
        )
        config["training_parameters"]["batch_size"] = trial.suggest_int(
            "batch_size", 8, 128
        )
        config["training_parameters"]["weight_decay"] = trial.suggest_float(
            "weight_decay", 1e-6, 1e-2
        )
        if network_type == "mlp":
            config["architecture_parameters"]["hidden_dim"] = trial.suggest_int(
                "hidden_dim", 64, 2048
            )
            config["architecture_parameters"]["hidden_dropout"] = (
                trial.suggest_float("hidden_dropout", 0.0, 1.0)
            )
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
    else:
        raise ValueError("Task not supported")

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

    network_type = config["architecture_parameters"]["network_type"]
    if network_type == "linear":
        config["architecture_parameters"]["input_dim"] = embeddings.shape[1]
        model = heads.LinearHead(config["architecture_parameters"])
    elif network_type == "mlp":
        config["architecture_parameters"]["input_dim"] = embeddings.shape[1]
        model = heads.MLP(config["architecture_parameters"])
    else:
        raise ValueError("Head type not supported")

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
    if on_ray_tuning:
        callbacks.append(
            PyTorchLightningPruningCallback(trial, monitor=f"val_loss")
        )
    callbacks.append(model.early_stopping() if not on_ray_tuning else model.early_stopping(patience))

    trainer = Trainer(
        default_root_dir=logger.base_dir,
        logger=lightning_logger,
        enable_checkpointing=False,
        max_epochs=epochs,
        enable_progress_bar=False,
        accumulate_grad_batches=model.gradient_accumulation_steps(),
        gradient_clip_val=model.gradient_clipping(),
        limit_train_batches=(model.epoch_sizing()),
        limit_val_batches=(model.epoch_sizing()),
        devices=devices,
        strategy=strategy,
        precision="16-mixed",
        callbacks=callbacks,
    )

    if on_ray_tuning:
        hyperparameters = dict(
            learning_rate=config["training_parameters"]["learning_rate"],
            batch_size=config["training_parameters"]["batch_size"],
            weight_decay=config["training_parameters"]["weight_decay"],
        )
        if network_type == "mlp":
            hyperparameters["hidden_dim"] = config["architecture_parameters"][
                "hidden_dim"
            ]
            hyperparameters["hidden_dropout"] = config["architecture_parameters"][
                "hidden_dropout"
            ]

        trainer.logger.log_hyperparams(hyperparameters)
    model.train()
    trainer.fit(model, data_loaders["train"], data_loaders["val"])

    if on_ray_tuning:
        callbacks[0].check_pruned()
        loss = trainer.callback_metrics[f"val_loss"].item()
        del trainer
        del data
        del data_loaders
        del scores
        del model
        gc.collect()
        return loss
    loss_plot = data_explore.create_loss_plot(
        json_path=f"{logger.base_dir}/{logger.experiment_name}_loss.json"
    )
    logger.save_plot(loss_plot, "training_validation_loss")

    trainer.test(
        model=model,
        ckpt_path=f"{logger.base_dir}/lightning_logs/best_model.ckpt",
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
    data,
    logger,
    split=None,
    num_workers=0,
    weights=None,
    sampler=False,
    n_trials=100,
):
    if version.parse(pl.__version__) < version.parse("2.2.1"):
        raise RuntimeError(
                "PyTorch Lightning>=2.2.1 is required for hyper-parameter tuning."
            )
    network_type = head_config["architecture_parameters"]["network_type"]

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
            data,
            logger,
            split,
            on_ray_tuning=True,
            num_workers=num_workers,
            weights=weights,
            sampler=sampler,
        ),
        n_trials=n_trials if network_type == "linear" else n_trials * 4,
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

    head_config["training_parameters"]["learning_rate"] = study.best_params["learning_rate"]
    head_config["training_parameters"]["batch_size"] = study.best_params["batch_size"]
    head_config["training_parameters"]["weight_decay"] = study.best_params["weight_decay"]
    if network_type == "mlp":
        head_config["architecture_parameters"]["hidden_dim"] = study.best_params["hidden_dim"]
        head_config["architecture_parameters"]["hidden_dropout"] = study.best_params[
            "hidden_dropout"
        ]

    return head_config
