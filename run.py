import os
import json
import argparse
from torch import set_float32_matmul_precision
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

import src.experiments
import src.datasets
import src.models

# These is for gpus with tensor cores
set_float32_matmul_precision('medium')

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

# Read config file
args = parser.parse_args()
with open(args.filename, 'r') as file:
    config = json.load(file)

# Retrieve classes from strings for torch_model, datamodule and lightningmodule
model_class = getattr(
    src.models,
    config['model_params']['name']
    )
datamodule_class = getattr(
    src.datasets,
    config['data_params']['datamodule_name']
    )
experiment_class = getattr(
    src.experiments,
    config['exp_params']['lightningmodule_name']
    )

# Initialize torch_model, datamodule and lightningmodule
model = model_class(**config['model_params'])
experiment = experiment_class(model, config['exp_params'])
data = datamodule_class(
    **config['data_params'], 
    pin_memory=len(config['trainer_params']['devices']) != 0
)

tb_logger =  TensorBoardLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['model_params']['name'],
)
tb_logger.log_hyperparams(config)

# For reproducibility
seed_everything(
    config['exp_params']['manual_seed'], 
    True
    )

runner = Trainer(
    logger=tb_logger,
    #log_every_n_steps=25,
    callbacks=[
        LearningRateMonitor(),
        ModelCheckpoint(
            save_top_k=2, 
            dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
            monitor= "val_loss",
            save_last= True),
        ],
    #strategy=DDPStrategy(find_unused_parameters=False),
    **config['trainer_params']
    )

print(f"======= Training {config['model_params']['name']} =======")
data.setup()
runner.fit(experiment, datamodule=data)