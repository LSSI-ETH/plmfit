import torch
from plmfit.shared_utils import utils
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from plmfit.models.lightning_model import LightningModel, PredictionWriter
from lightning.pytorch.strategies import DeepSpeedStrategy
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from lightning.pytorch.tuner import Tuner

def extract_embeddings(args, logger):
    # head_config = utils.load_config(f"inferring/{args.head_config}")
    # task = head_config["architecture_parameters"]["task"]

    # Load dataset
    data = utils.load_dataset(args.data_type)

    model = utils.init_plm(args.plm, logger, task="extract_embeddings")
    assert model != None, "Model is not initialized"

    model.experimenting = (
        args.experimenting == "True"
    )  # If we are in experimenting mode

    model.set_layer_to_use(args.layer)
    model.py_model.reduction = args.reduction

    encs = model.categorical_encode(data)
    encs = torch.tensor(encs)

    logger.save_data(vars(args), "arguments")

    # TODO: Create predict dataloaders
    data_loader = utils.create_predict_data_loader(encs)

    model = LightningModel(
        model.py_model,
        plmfit_logger=logger,
        log_interval=100,
        experimenting=model.experimenting,
        train=False,
    )
    model.eval()
    lightning_logger = TensorBoardLogger(
        save_dir=logger.base_dir, version=0, name="lightning_logs"
    )

    strategy = DeepSpeedStrategy(
        stage=3,
        offload_optimizer=True,
        offload_parameters=True,
        load_full_weights=True,
    )

    devices = args.gpus if torch.cuda.is_available() else 1
    strategy = strategy if torch.cuda.is_available() else "auto"

    pred_writer = PredictionWriter(logger=logger, write_interval="epoch", split_size=args.split_size)

    trainer = Trainer(
        default_root_dir=logger.base_dir,
        logger=lightning_logger,
        enable_progress_bar=False,
        devices=devices,
        strategy=strategy,
        precision="16-mixed",
        callbacks=[pred_writer],
    )
    # tuner = Tuner(trainer)

    # # Auto-scale batch size by growing it exponentially (default)
    # tuner.scale_batch_size(model, mode="power")

    if torch.cuda.is_available():
        estimate_zero3_model_states_mem_needs_all_live(
            model, num_gpus_per_node=int(args.gpus), num_nodes=1
        )

    trainer.predict(model=model, dataloaders=data_loader)
