import os
import json
import argparse
import numpy as np
from pathlib import Path
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from src.experiments import BaseExperiment
from src.datasets import XorDataModule
from src.models import all_models

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

model = all_models[config['model_params']['name']](**config['model_params'])

experiment = BaseExperiment(
    model,
    config['exp_params']
    )

data = XorDataModule(
    **config["data_params"], 
    pin_memory=len(config['trainer_params']['gpus']) != 0
    )

data.setup()
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
    strategy=DDPPlugin(find_unused_parameters=False),
    **config['trainer_params']
    )

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)