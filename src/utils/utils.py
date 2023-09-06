import pytorch_lightning as pl
## Utils to handle newer PyTorch Lightning changes from version 0.6
def data_loader(fn):
    """
    Decorator to handle the deprecation of data_loader from 0.7
    :param fn: User defined data loader function
    :return: A wrapper for the data_loader function
    """
    def func_wrapper(self):
        try: # Works for version 0.6.0
            return pl.data_loader(fn)(self)

        except: # Works for version > 0.6.0
            return fn(self)
    return func_wrapper


import yaml
import torch
import random
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from dataset import MazeDataset
from models.beta_vae import BetaVAE
#from models.beta_vae_MLP import BetaVAE_MLP
from models.beta_vae_CLF import BetaVAE_CLF

models = {
    'BetaVAE': BetaVAE, 
    #'BetaVAE_MLP': BetaVAE_MLP, 
    'BetaVAE_CLF': BetaVAE_CLF,
}

def load_model_from_checkpoint(ckp_path, config):
    # load pytorch model checkpoint from file
    model_type = config['model_params']['name']
    model = models[model_type](**config['model_params'])
    if not torch.cuda.is_available():
        map_location=torch.device('cpu')
    else:
        map_location=None
    state = torch.load(ckp_path, map_location=map_location)['state_dict']
    # remove 'model.' prefix from state dict keys
    state = {k[6:]: v for k, v in state.items()} 
    model.load_state_dict(state, strict=0) 
    model = model.eval()
    return model

def load_config(config_path):
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def load_dataset(config):
    path2imgs = config['data_params']['data_path']
    imgs = pd.read_csv(path2imgs, header=None).values
    dataset = MazeDataset(imgs, config["data_params"]["num_labels"])
    return dataset

def load_experiment(config_path, ckp_path, data_path=None):
    config = load_config(config_path)
    model = load_model_from_checkpoint(ckp_path, config)
    if data_path is not None:
        config['data_params']['data_path'] = data_path
    dataset = load_dataset(config)
    return dataset, model, config

def plot_random_sample(dataset, model, model_type="conv", idx=None):
    """
    model_type: 'conv' or 'fully_connected'
    """
    if idx is None:
        idx = random.choice(range(len(dataset)-15000, len(dataset)))

    in_ = dataset[idx:idx+1][0]
    labels = dataset[idx:idx+1][1]

    if model_type == 'conv':
        in_rec = model.generate(in_, labels=labels).squeeze().detach().numpy() # for BVAE
    elif model_type == 'fully_connected':
        in_rec = model.generate(in_.reshape(-1,169)).reshape(13,13).detach().numpy() # for BVAE_MLP

    # use this with mnist
    # in_ = d[idx][0]
    # in_rec = model.generate(in_.reshape(1,*in_.shape)).squeeze().detach().numpy()

    in_rec_thrd = np.where(in_rec > 0.5, 1, 0)
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(in_.squeeze().detach().numpy(), cmap='binary')
    axs[0].set_title('Sample')

    axs[1].imshow(in_rec, cmap='binary')
    axs[1].set_title('Reconstruction')

    for ax in axs:
        ax.set_yticks([])
        ax.set_xticks(range(0,13,2))

    fig.show()
    
    print(f'Sample idx :  {idx}')
    print(f'Sample labels :  {labels}')

    return fig, axs