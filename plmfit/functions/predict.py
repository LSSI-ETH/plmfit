import torch
from plmfit.shared_utils import utils
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from plmfit.models.lightning_model import LightningModel, PredictionWriter
from lightning.pytorch.strategies import DeepSpeedStrategy
import json
from plmfit.models.fine_tuners import (
    FullRetrainFineTuner,
    LowRankAdaptationFineTuner,
    BottleneckAdaptersFineTuner,
)
import plmfit.models.downstream_heads as heads

def predict(args, logger):

    # Load dataset
    input_data = utils.load_dataset(args.prediction_data)

    model_metadata = json.load(open(f"{args.model_metadata}", "r"))

    model_path = args.model_path

    # If model is a fine-tuned PLM
    if (
        model_metadata["arguments"]["function"] == "fine_tuning"
        and model_metadata["arguments"]["ft_method"] != "feature_extraction"
    ):
        # Initialize model
        model = utils.init_plm(
            model_metadata["arguments"]["plm"],
            logger,
            task=model_metadata["head_config"]["architecture_parameters"]["task"],
        )
        assert model != None, "Model is not initialized"
        model.set_layer_to_use(model_metadata["arguments"]["layer"])

        # If model outputs logits, turn them into probabilities
        if "logits" in model_metadata["head_config"]["training_parameters"]["loss_f"]:
            model_metadata["head_config"]["architecture_parameters"][
                "output_activation"
            ] = "sigmoid"

        head = heads.init_head(
            config=model_metadata["head_config"], input_dim=model.emb_layers_dim
        )

        model.py_model.set_head(head)
        model.py_model.reduction = model_metadata["arguments"]["reduction"]

        if model_metadata["arguments"]["ft_method"] == "lora":
            fine_tuner = LowRankAdaptationFineTuner(
                training_config=model_metadata["head_config"][
                    "training_parameters"
                ],
                logger=logger,
            )
        elif model_metadata["arguments"]["ft_method"] == "bottleneck_adapters":
            fine_tuner = BottleneckAdaptersFineTuner(
                training_config=model_metadata["head_config"]["training_parameters"],
                logger=logger,
            )
        elif model_metadata["arguments"]["ft_method"] == "full":
            fine_tuner = FullRetrainFineTuner(
                training_config=model_metadata["head_config"]["training_parameters"],
                logger=logger,
            )
        else:
            raise ValueError("Fine Tuning method not supported")

        model = fine_tuner.prepare_model(
            model, target_layers=model_metadata["arguments"]["target_layers"]
        )

        model.py_model.task = model_metadata["head_config"]["architecture_parameters"][
            "task"
        ]

        strategy = DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            load_full_weights=True,
        )

        # Encode input data
        encoded_input_data = model.categorical_encode(input_data)
        data_loader = utils.create_predict_data_loader(
            encoded_input_data, batch_size=args.batch_size
        )

        model = model.py_model

    # If model is a feature-extractor
    elif (
        model_metadata["arguments"]["function"] == "fine_tuning"
        and model_metadata["arguments"]["ft_method"] == "feature_extraction"
    ):
        # If model outputs logits, turn them into probabilities
        if "logits" in model_metadata["head_config"]["training_parameters"]["loss_f"]:
            model_metadata["head_config"]["architecture_parameters"][
                "output_activation"
            ] = "sigmoid"

        model = heads.init_head(
            config=model_metadata["head_config"],
            input_dim=model_metadata["head_config"]["architecture_parameters"][
                "input_dim"
            ],
        )

        strategy = "auto"

        encoded_input_data = input_data
        data_loader = utils.create_predict_data_loader(
            encoded_input_data, batch_size=args.batch_size
        )

    # If model is one-hot encoder
    else:
        # If model outputs logits, turn them into probabilities
        if "logits" in model_metadata["head_config"]["training_parameters"]["loss_f"]:
            model_metadata["head_config"]["architecture_parameters"]["output_activation"] = "sigmoid"

        model = heads.init_head(
            config=model_metadata["head_config"],
            input_dim=model_metadata["head_config"]["architecture_parameters"][
                "input_dim"
            ],
        )
        strategy = "auto"

        try:
            max_len = max(input_data["len"].values)
        except:
            max_len = max(len(seq) for seq in input_data["aa_seq"])

        tokenizer = utils.load_tokenizer("proteinbert")  # Use same tokenizer as proteinbert
        num_classes = tokenizer.get_vocab_size(with_added_tokens=False)
        encoded_input_data = utils.categorical_encode(
            input_data["aa_seq"].values,
            tokenizer,
            max_len,
            logger=logger,
            model_name="proteinbert",
        )
        data_loader = utils.create_predict_data_loader(
            encoded_input_data, batch_size=args.batch_size, dataset_type="one_hot"
        )
        data_loader.dataset.set_num_classes(num_classes)
        if (
            model_metadata["head_config"]["architecture_parameters"]["task"]
            == "token_classification"
        ):
            data_loader.dataset.set_flatten(False)

    model.eval()

    model = LightningModel(
        model,
        model_metadata["head_config"]["training_parameters"],
        plmfit_logger=logger,
        log_interval=100,
    )
    lightning_logger = TensorBoardLogger(
        save_dir=logger.base_dir, version=0, name="lightning_logs"
    )

    devices = args.gpus if torch.cuda.is_available() else 1
    strategy = strategy if torch.cuda.is_available() else "auto"

    pred_writer = PredictionWriter(
        logger=logger, write_interval="epoch", split_size=args.split_size, format="csv"
    )

    trainer = Trainer(
        default_root_dir=logger.base_dir,
        logger=lightning_logger,
        enable_progress_bar=False,
        devices=devices,
        strategy=strategy,
        precision="16-mixed",
        callbacks=[pred_writer],
        inference_mode=True,
    )

    trainer.predict(model=model, ckpt_path=model_path, dataloaders=data_loader)
