import torch
from plmfit.shared_utils import utils
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from plmfit.models.lightning_model import LightningModel
from lightning.pytorch.strategies import DeepSpeedStrategy

def predict(args, logger):
    lightning_logger = TensorBoardLogger(save_dir=logger.base_dir, version=0, name="lightning_logs")

    strategy = DeepSpeedStrategy(
        stage=3,
        offload_optimizer=True,
        offload_parameters=True,
        load_full_weights = True,
    )

    devices = args.gpus if torch.cuda.is_available() else 1
    strategy = strategy if torch.cuda.is_available() else 'auto'

    trainer = Trainer(
        default_root_dir=logger.base_dir,
        logger=lightning_logger, 
        enable_progress_bar=False,
        devices=devices,
        strategy=strategy,
        precision="16-mixed"
    )
    
    # Load the model from checkpoint
    model = LightningModel.load_from_checkpoint(checkpoint_path=f'{logger.base_dir}/best_model.ckpt')

    # Assuming data_loaders['test'] is prepared elsewhere and available here
    trainer.predict(model=model, dataloaders=data_loaders['test'])