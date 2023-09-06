import numpy as np
import torch
from sklearn.metrics import f1_score
from models.base import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class BetaVAE_CLF(BaseVAE):

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
                 clf_task_num:int = 0,
                 **kwargs) -> None:
        super(BetaVAE_CLF, self).__init__()

        clf_tasks_dict = {
            0 : self.map_label2idx_task0,
            1 : self.map_label2idx_task1,
            2 : self.map_label2idx_task2,
            3 : self.map_label2idx_task3,
        }

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.clf_task_num = clf_task_num
        self.clf_task = clf_tasks_dict[clf_task_num]

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
        

        # Build Classifier
        if self.clf_task_num == 0:
            self.clf = nn.Sequential(
                nn.Linear(latent_dim, 1500),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(0.25),
                nn.Linear(1500, 25),
                nn.Softmax(dim=1)
            )
        elif self.clf_task_num in [1, 2, 3]:
            self.clf = nn.Sequential(
                nn.Linear(latent_dim, 1024),
                nn.Linear(1024, 1),
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
        #result = result.view(-1, 1152)
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

    def classify(self, z: Tensor) -> Tensor:
        preds = self.clf(z)
        return preds
    
    # @staticmethod
    # def get_new_labels_task0(label):
    #     """
    #     ___|___|_|___|___|
    #     """
    #     if label <= 6 or label == 12:
    #         return label//3
    #     else:
    #         return label//3 + 1
    
    @staticmethod
    def get_new_labels_task0(label):
        if label in [0,1]:
            return 0
        elif label in [2,3,4]:
            return 1
        elif label in [5,6,7]:
            return 2
        elif label in [8,9,10]:
            return 3
        elif label in [11,12]:
            return 4
    
    def map_label2idx_task0(self, labels: Tensor) -> Tensor:
        device = labels.device
        labels = labels.cpu()
        labels = np.vectorize(self.get_new_labels_task0)(labels)

        if len(labels.shape) == 1:
            res = labels[0]*5 + labels[1]
            return torch.tensor(res, device=device, dtype=torch.long)
        else:
            res = labels[:,0]*5 + labels[:,1]
            return torch.tensor(res, device=device, dtype=torch.long)
        
    # @staticmethod
    # def map_label2idx_task0(labels: Tensor) -> Tensor:
    #     if len(labels.shape) == 1:
    #         res = labels[0]//3*4 + labels[1]//3
    #         return res
    #     else:
    #         res = labels[:,0]//3*4 + labels[:,1]//3
    #         return res

    @staticmethod
    def map_label2idx_task1(labels: Tensor) -> Tensor:
        device = labels.device
        labels = labels.cpu()
        if len(labels.shape) == 1:
            res = np.where(labels[0]==labels[1],1,0).reshape(-1,1)
            return torch.tensor(res, device=device, dtype=torch.float)
        else:
            res = np.where(labels[:,0]==labels[:,1],1,0).reshape(-1,1)
            return torch.tensor(res, device=device, dtype=torch.float)

    @staticmethod
    def map_label2idx_task2(labels: Tensor) -> Tensor:
        device = labels.device
        labels = labels.cpu()
        if len(labels.shape) == 1:
            res = np.where(labels[0]<=labels[1],1,0).reshape(-1,1)
            return torch.tensor(res, device=device, dtype=torch.float)
        else:
            res = np.where(labels[:,0]<=labels[:,1],1,0).reshape(-1,1)
            return torch.tensor(res, device=device, dtype=torch.float)

    @staticmethod
    def map_label2idx_task3(labels: Tensor) -> Tensor:
        device = labels.device
        labels = labels.cpu()
        if len(labels.shape) == 1:
            condition = (labels[0] < 6) | ((labels[0] >= 6) & (labels[1] >= 6))
        else:
            condition = (labels[:,0] < 6) | ((labels[:,0] >= 6) & (labels[:,1] >= 6)) 
        res = np.where(condition,1,0).reshape(-1,1)
        return torch.tensor(res, device=device, dtype=torch.float)

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        labels = kwargs['labels']
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        preds = self.classify(z)
        recons = self.decode(z)
        return  [recons, input, mu, log_var, preds, labels]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons, input, mu, log_var, preds, labels = args

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

        # Compute classification loss
        #print(preds.shape, labels.shape)
        # the weight of the classification loss is controlled by the parameter clf_w, that depend on self.C_stop_iter
        
        # if self.num_iter > 4000:
        #     # clf_w = torch.clamp(
        #     #     torch.Tensor([2e-3 * (self.num_iter-4000),]).to(input.device), 0,  
        #     #     100
        #     # )
        #     clf_w = 1
        # else:
        #     clf_w = 0

        true_labels = self.clf_task(labels)
        # clf_w = torch.clamp(torch.tensor([self.num_iter / self.C_stop_iter]).to(input.device), 0, 1)
        clf_w = torch.clamp(torch.tensor([(self.num_iter - self.C_stop_iter) / self.C_stop_iter]).to(input.device), 0, 1)

        # if self.num_iter < 1500:
        #     clf_w = clf_w/4
        
        if self.clf_task_num == 0:
            clf_loss = 10 * clf_w * F.cross_entropy(preds, true_labels)
            pred_labels = torch.argmax(preds, dim=1)
            f1 = f1_score(true_labels.cpu(), pred_labels.cpu(), average='macro')
        
        elif self.clf_task_num in [1,2,3]:
            # if self.num_iter < self.C_stop_iter:
            #     clf_w = 0
            # else:
            #     clf_w = 1
            clf_loss = clf_w * F.binary_cross_entropy(preds, true_labels)
            pred_labels =  np.where(preds.cpu()>=.5, 1, 0)
            f1 = f1_score(true_labels.cpu(), pred_labels)

        ##clf_loss = F.cross_entropy(preds, labels.squeeze())

        # Compute total loss
        loss = betavae_loss + clf_loss

        return {'loss':loss, 'betavae_loss': betavae_loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss, 'clf_loss':clf_loss, 'f1_score': f1}

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