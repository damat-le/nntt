import torch
from models.base import BaseVAE
from torch import nn
from torch import bernoulli
from torch.nn import functional as F
from utils.types_ import *

class BetaVAE(BaseVAE):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 **kwargs) -> None:
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=32,
                kernel_size=4, 
                stride=2, 
                padding=1
                ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32, 
                out_channels=64,
                kernel_size=3, 
                stride=2, 
                padding=1
                ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64, 
                out_channels=256,
                kernel_size=3, 
                stride=2, 
                padding=1
                ),
            nn.LeakyReLU(),

        )
        new_dim = 256*(2**2)
        self.fc_mu = nn.Linear(new_dim, latent_dim)
        self.fc_var = nn.Linear(new_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, new_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,

                ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=in_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=1,
                ),
            nn.Sigmoid()
        )
        
    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        #print('ENCODER output', result.shape)
        result = torch.flatten(result, start_dim=1)
        #print('ENCODER output flattened', result.shape)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        #print('MU', mu.shape)
        #print('LOG_VAR', log_var.shape)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        #print('Z', z.shape)
        #z = z.view(-1, self.latent_dim, 1, 1)
        #print('Z reshaped', z.shape)
        result = self.decoder_input(z)
        #print('DECODER input', result.shape)
        result = result.view(-1, 256, 2, 2)
        result = self.decoder(result)
        #print('DECODER', result.shape)
        #result = result.view(-1, 845)
        #result = self.final_layer(result)
        #result = result.view(-1, 1, 13, 13)
        #print('FINAL', result.shape)
        #result = bernoulli(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        labels = kwargs['labels']
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        return  [recons, input, mu, log_var, labels]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons, input, mu, log_var, labels = args

        # Compute beta-VAE loss
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            betavae_loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            if self.num_iter < 1500:
                C = C/4
            betavae_loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        # Compute total loss
        loss = betavae_loss 

        return {'loss':loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x, **kwargs)[0]