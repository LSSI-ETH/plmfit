import torch
import plmfit.models.downstream_heads as heads
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from plmfit.models.lightning_model import LightningModel
from lightning.pytorch.strategies import DeepSpeedStrategy
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import lightning.pytorch as pl
from optuna.integration import PyTorchLightningPruningCallback
from packaging import version
from optuna.visualization import plot_optimization_history, plot_slice
from plmfit.shared_utils import utils, data_explore
from plmfit.logger import LogOptunaTrialCallback
import copy


def onehot(args, logger):
    head_config = utils.load_config(f"training/{args.head_config}")
    task = head_config["architecture_parameters"]["task"]

    # Load dataset
    data = utils.load_dataset(args.data_type)
    # data = data[:10000]

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
    max_len = max(data["len"].values)
    if args.evaluate == "True" and split is None:
        raise ValueError("Cannot evaluate without a standard testing split")

    tokenizer = utils.load_tokenizer("proteinbert")  # Use same tokenizer as proteinbert
    num_classes = tokenizer.get_vocab_size(with_added_tokens=False)
    encs = utils.categorical_encode(
        data["aa_seq"].values,
        tokenizer,
        max_len,
        logger=logger,
        model_name="proteinbert",
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
        )
    else:
        raise ValueError("Task not supported")

    if args.ray_tuning == "True":
        assert args.evaluate == "False", "Cannot evaluate and tune at the same time"

        head_config = hyperparameter_tuning(
            task,
            args,
            head_config,
            encs,
            scores,
            logger,
            split=split,
            num_workers=0,
            weights=weights,
            sampler=sampler,
            n_trials=100,
            num_classes=num_classes,
        )

    logger.save_data(vars(args), "arguments")
    logger.save_data(head_config, "head_config")

    objective(
        None,
        task,
        args,
        head_config,
        encs,
        scores,
        logger,
        split=split,
        on_ray_tuning=False,
        num_workers=0,
        weights=weights,
        sampler=sampler,
        num_classes=num_classes,
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
    num_classes=21,
):
    config = copy.deepcopy(head_config)

    network_type = config["architecture_parameters"]["network_type"]

    epochs = config["training_parameters"]["epochs"]
    if trial is not None and on_ray_tuning:
        epochs = config["training_parameters"]["epochs"] // 5
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
        if network_type == "rnn":
            config["architecture_parameters"]["hidden_dim"] = trial.suggest_int(
                "hidden_dim", 16, 512
            )
            config["architecture_parameters"]["dropout"] = trial.suggest_float(
                "dropout", 0.0, 1.0
            )

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
        dataset_type="one_hot",
    )

    data_loaders["train"].dataset.set_num_classes(num_classes)
    data_loaders["val"].dataset.set_num_classes(num_classes)
    data_loaders["test"].dataset.set_num_classes(num_classes)
    if task == "token_classification":
        data_loaders["train"].dataset.set_flatten(False)
        data_loaders["val"].dataset.set_flatten(False)
        data_loaders["test"].dataset.set_flatten(False)

    network_type = config["architecture_parameters"]["network_type"]
    if network_type == "linear":
        config["architecture_parameters"]["input_dim"] = (
            embeddings.shape[1] * num_classes
            if task != "token_classification"
            else num_classes
        )  # Account for one-hot encodings
        model = heads.LinearHead(config["architecture_parameters"])
    elif network_type == "mlp":
        config["architecture_parameters"]["input_dim"] = (
            embeddings.shape[1] * num_classes
            if task != "token_classification"
            else num_classes
        )  # Account for one-hot encodings
        model = heads.MLP(config["architecture_parameters"])
    elif network_type == "rnn":
        config["architecture_parameters"]["input_dim"] = (
            embeddings.shape[1] * num_classes
            if task != "token_classification"
            else num_classes
        )  # Account for one-hot encodings
        model = heads.RNN(config["architecture_parameters"])
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
    epoch_sizing = model.epoch_sizing()
    if on_ray_tuning:
        epoch_sizing = 0.1
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
        if network_type == "rnn":
            hyperparameters["hidden_dim"] = config["architecture_parameters"][
                "hidden_dim"
            ]
            hyperparameters["dropout"] = config["architecture_parameters"]["dropout"]

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
        ckpt_path=f"{logger.base_dir}/lightning_logs/best_model.ckpt"
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
    n_trials=100,
    num_classes=21,
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
            scores,
            logger,
            split,
            on_ray_tuning=True,
            num_workers=num_workers,
            weights=weights,
            sampler=sampler,
            num_classes=num_classes,
        ),
        n_trials=n_trials if network_type == "linear" else n_trials * 3,
        callbacks=[LogOptunaTrialCallback(logger)],
        n_jobs = int(args.gpus),
        gc_after_trial = True,
        catch=(FileNotFoundError,),
    )
    logger.mute = False
    history = plot_optimization_history(study)
    slice = plot_slice(study)
    history.write_image(f"{logger.base_dir}/optuna_optimization_history.png")
    slice.write_image(f"{logger.base_dir}/optuna_slice.png")

    head_config["training_parameters"]["learning_rate"] = study.best_params[
        "learning_rate"
    ]
    head_config["training_parameters"]["batch_size"] = study.best_params["batch_size"]
    head_config["training_parameters"]["weight_decay"] = study.best_params[
        "weight_decay"
    ]
    if network_type == "mlp":
        head_config["architecture_parameters"]["hidden_dim"] = study.best_params["hidden_dim"]
        head_config["architecture_parameters"]["hidden_dropout"] = study.best_params["hidden_dropout"]
    if network_type == "rnn":
        head_config["architecture_parameters"]["hidden_dim"] = study.best_params["hidden_dim"]
        head_config["architecture_parameters"]["dropout"] = study.best_params["dropout"]

    return head_config
