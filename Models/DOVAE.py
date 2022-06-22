from typing import List, Tuple
import torch
from torch import nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

from DOBase import DOBase

class DO(DOBase):
    def __init__(self, input_size : int, hidden_size : int):
        super().__init__()
        self.mean_layer = None
        self.logvar_layer =None
        self.decoder = None
        self.input_size = input_size
        self.hidden_size = hidden_size
    
    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.mean_layer == None or self.logvar_layer == None:
            return x, None
        return [self.mean_layer(x), self.logvar_layer(x)]
    
    def reparameterize(self, mu : torch.Tensor, logvar : torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return [self.decoder(z), mu, logvar, z]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def vary(self, x: torch.Tensor, layer : int, encode : bool) -> torch.Tensor:
        hidden_repr, logvar = self.encode(x, layer)
        new_hidden_repr = hidden_repr.clone().detach()
        std = torch.exp(0.5 * logvar)
        # Stepping with 5% of the variables in the latent space
        changed_indices = max(round(0.05 * hidden_repr.shape[1]), 1)

        # hillclimb
        if layer == 0:
            i = torch.randint(0,hidden_repr.shape[1], (hidden_repr.shape[0],))
            new_hidden_repr[torch.arange(hidden_repr.shape[0]),i] *= -1
        else:
            i = torch.randint(0,hidden_repr.shape[1], (changed_indices,hidden_repr.shape[0]))
            # Provides values of either 1 or -1
            directions = torch.randint(0,2,i.shape,dtype=torch.float32) * 2 - 1
            steps = std[torch.arange(hidden_repr.shape[0]),i] * 5 * directions
            new_hidden_repr[torch.arange(hidden_repr.shape[0]),i] += steps

        old_reconstruction = torch.sign(self.decode(hidden_repr, layer))
        new_reconstruction = torch.sign(self.decode(new_hidden_repr, layer))
        delta_s = new_reconstruction - old_reconstruction

        new_solution = x + delta_s
        return new_solution
    
    def transition(self) -> None:
        self.mean_layer = weight_norm(nn.Linear(self.input_size, self.hidden_size), name='weight')
        self.std_layer = weight_norm(nn.Linear(self.input_size, self.hidden_size), name='weight')      
        decoder_layer = weight_norm(nn.Linear(self.hidden_size, self.input_size), name='weight')
        self.decoder = nn.Sequential(decoder_layer,nn.Tanh())
    
    def loss(self, x: torch.Tensor, recon_x : torch.Tensor,
             mu : torch.Tensor, logvar : torch.Tensor, beta : int) -> dict:
        recon_MSE = F.mse_loss(recon_x, x)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # (This has been changed from sum to mean to scale it properly)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_MSE + beta * KLD
        return {"loss" : loss, "recon" : recon_MSE, "kld" : KLD}