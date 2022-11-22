import torch
from torch import nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

from Models.DOBase import DOBase

class DOVAE(DOBase):
    """
    Implements a VAE model for DO. So far, only a single-layered VAE has been used,
    which is implemented here.
    """
    def __init__(self, input_size: int, hidden_size: int, device: torch.device):
        """
        Constructor method for the VAE. All layers of the model start off as empty, and
        will be set and reset during the transition method.

        Args:   
            input_size: int
                The size of each problem solution.
            hidden_size: int
                The dimensions of the latent space.
            device: torch.device
                The device the model is loaded onto.
        """
        super().__init__()
        self.mean_layer = None
        self.logvar_layer =None
        self.decoder = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
    
    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Encodes points into the solution space into a latent distribution.

        Args:
            x: torch.Tensor
                The input of shape n x W, where n is the number of solutions being passed
                through and W is the size of each solution.
        
        Returns:
            A list containing the means and log-variances of the latent distributions, each 
            of shape n x L where L is the size of the latent dimension. 
        """
        if self.mean_layer == None or self.logvar_layer == None:
            return x, None
        return [self.mean_layer(x), self.logvar_layer(x)]
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Samples a latent distribution in such a way that gradients can flow through the 
        entirety of the network, as detailed in "Auto-Encoding Variational Bayes", 
        D. Kingma, M. Welling, 2013.

        Args:
            mu: torch.Tensor
                The means of the latent distributions of shape n x L where L is the size
                of the latent dimension and n is the number of points.
            logvar: torch.Tensor
                The log-variances of the latent distribution of shape n x L.
        
        Returns:
            Points sampled from each latent distribution, of shape n x L.
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std, device=self.device)
        return mu + eps*std

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Passes solutions through the network to produce an output.

        Args:
            x: torch.Tensor
                The input of shape n x W, where n is the number of solutions being passed
                through and W is the size of each solution.
        
        Returns:
            A list containing the model reconstruction (shape n x W), and the mean, log-variance,
            and sampled point of each latent distribution (each of shape n x L, where L is the
            size of the latent space).
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return [self.decode(z), mu, logvar, z]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes latent points sampled from the latent distibutions.

        Args:
            z: torch.Tensor
                The latent points to be decoded, of shape n x L where n is the number of latent
                points and L is the size of the latent space.

        Returns:
            The latent points decoded back into the solution space, of size n x W where W is the 
            size of the solution space.
        """
        return self.decoder(z)
    
    def vary(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Model-Informed Variation to solutions.

        Args:
            x: torch.Tensor
                The solutions to be varied, represented as a tensor of size N x W where N is the 
                number of solutions and W is the size of each solution.

        Returns:
            The new solutions after variation has been applied, of size N x W.
        """
        hidden_repr, logvar = self.encode(x)
        new_hidden_repr = hidden_repr.clone().detach()
        std = torch.exp(0.5 * logvar)
        # Stepping with 5% of the variables in the latent space
        changed_indices = max(round(0.05 * hidden_repr.shape[1]), 1)

        i = torch.randint(0,hidden_repr.shape[1], (changed_indices,hidden_repr.shape[0]))
        # Provides values of either 1 or -1
        directions = torch.randint(0,2,i.shape,dtype=torch.float32, device=self.device) * 2 - 1
        # Take steps of size 5*std
        steps = std[torch.arange(hidden_repr.shape[0]),i] * 5 * directions
        new_hidden_repr[torch.arange(hidden_repr.shape[0]),i] += steps

        old_reconstruction = torch.sign(self.decode(hidden_repr))
        new_reconstruction = torch.sign(self.decode(new_hidden_repr))
        delta_s = new_reconstruction - old_reconstruction

        new_solution = torch.sign(x + delta_s)
        return new_solution
    
    def transition(self) -> None:
        """
        Resets the parameters of the VAE to learn the newly optimised solutions. This should be 
        called after Model-Informed Variation has been applied to the solutions.
        """
        self.mean_layer = weight_norm(nn.Linear(self.input_size, self.hidden_size, device=self.device), name='weight')
        self.logvar_layer = weight_norm(nn.Linear(self.input_size, self.hidden_size, device=self.device), name='weight')
              
        decoder_layer = weight_norm(nn.Linear(self.hidden_size, self.input_size), name='weight')
        self.decoder = nn.Sequential(decoder_layer,nn.Tanh()).to(device=self.device)
    
    def loss(self, x: torch.Tensor, recon: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor, beta: float) -> dict:
        """
        Calculates the loss function. This is done by adding the MSE of the input and the 
        KL Divergence between the latent distribution and unit normal distribution. 

        Args:
            x: torch.Tensor
                The input to the model of size n x W, where n is the size of the batch and W is 
                the size of each solution.
            recon: torch.Tensor
                The reconstruction of x that the model outputs.
            mu: torch.Tensor
                The means of the latent distributions of shape n x L where L is the size
                of the latent dimension and n is the number of points.
            logvar: torch.Tensor
                The log-variances of the latent distribution of shape n x L.
            beta: float
                The coefficient of the KL Divergence term in the loss function

        Returns:
            The loss dictionary containing the total loss, reconstruction error and KL Divergence.
        """
        recon_MSE = F.mse_loss(recon, x)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # (This has been changed from sum to mean to scale it properly)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_MSE + beta * KLD
        return {"loss" : loss, "recon" : recon_MSE, "kld" : KLD}
    
    def learn_from_sample(self, x: torch.Tensor, optimizer: torch.optim.Optimizer,
                          beta: float) -> dict:
        """
        Handles learning from a sample of solutions.

        Args:
            x: torch.Tensor
                The sample of solutions to learn from.
            optimizer: torch.optim.Optimizer
                The optimizer that handles the adjustment of weights.
            beta: float
                The coefficient of theKL Divergence term.

        Returns:
            The loss dictionary containing the total loss, reconstruction error and KL Divergence. 
        """
        self.train()
        output, mu, logvar, z = self.forward(x)
        loss_dict = self.loss(x, output, mu, logvar, beta)
        loss = loss_dict["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss_dict