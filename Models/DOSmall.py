from typing import List
import torch
import torch.nn.functional as F

from torch import nn

from .DOBase import DOBase

class DOSmall(DOBase):
    """
    Implements the AE model in DO with a small latent space (compression < 0.1).
    """
    def __init__(self, input_size: int):
        """
        Constructor method for the AE. The encoder and decoder start of as empty models, as 
        layers get added to them during subsequent transitions.

        Args:
            input_size: int
                The size of the solutions in a given combinatorial optimisation problem.
            dropout_prob: float
                The amount of dropout that occurs in the input to the model.
        """
        super().__init__()
        self.encoder = None
        self.decoder = None
        self.pos_emb = None
        self.input_size = input_size   
        self.num_layers = 1
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Passes the input through a deterministic decoder.

        Args:
            x: torch.Tensor
                The input to be passed through the model. This is of shape n x W, where n is
                the number of solutions being passed through and W is the size of each solution.

        Returns:
            A list containing the reconstruction of the input of size n x W, and the latent point
            calculated from the input, of size n x L where L is the size of the deepest latent
            space.
        """
        z = self.encoder(x)
        c = self.cosine_similarity(z, self.pos_emb)
        return [self.decoder(c), z]
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input up to a specified layer in the model.

        Args:
            x: torch.Tensor
                The input of shape n x W, where n is the number of solutions being passed
                through and W is the size of each solution.
            layer: int
                The deepest layer the latent space is calculated up to.
        
        Returns:
            The generated latent space at the specified layer of size n x L where L is the size
            of the latent space at that layer.
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes latent points starting from a specified layer.

        Args:
            z: torch.Tensor
                The latent points to be decoded, of shape n x L where n is the number of latent
                points and L is the size of the latent space at the specified layer.
            layer: int
                The layer at which the latent points lie.

        Returns:
            The latent points decoded back into the solution space, of size n x W where W is the 
            size of the solution space.
        """
        c = self.cosine_similarity(z, self.pos_emb)
        return self.decoder(c)
    
    def pos_step(self, z: torch.Tensor) -> torch.Tensor:
        """
        Calculates the step taken in the latent space according to the 'Encode' method specified
        in Jamie Caldwell's thesis.

        Args:
            x: torch.Tensor
                The solutions of size N x W, where n is the number of solutions and W is the size
                of each solution.
            layer: int
                The layer at which the solutions are being encoded into.

        Returns:
            The step sizes to be taken in the latent space, represented as a tensor of shape 
            N x L, where L is the size of the latent space at the specified layer.
        """
        # Select a pos vector per solution
        i = torch.randint(0, self.pos_emb.shape[0], (z.shape[0],))
        direction = self.pos_emb[i] - z
        direction = F.dropout(direction, 0.5)
        return direction * torch.rand((direction.shape[0],1))

    
    def vary(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Model-Informed Variation to solutions at a given layer.

        Args:
            x: torch.Tensor
                The solutions to be varied, represented as a tensor of size N x W where N is the 
                number of solutions and W is the size of each solution.
            layer: int
                The layer at which Model-Informed Variation will be applied.
            encode: bool
                A flag indicating whether the Encode method of variation shall be used. If False,
                the Assign method will be used instead.

        Returns:
            The new solutions after variation has been applied, of size N x W.
        """
        hidden_repr = self.encode(x)
        new_hidden_repr = hidden_repr.clone().detach()

        d_h = self.pos_step(new_hidden_repr)
        new_hidden_repr += d_h

        old_reconstruction = torch.sign(self.decode(hidden_repr))
        new_reconstruction = torch.sign(self.decode(new_hidden_repr))
        delta_s = new_reconstruction - old_reconstruction

        new_solution = torch.sign(x + delta_s)
        return new_solution
    
    def transition(self, hidden_size : int, points_number: int) -> None:
        """
        Adds a new layer to the model. This should be called after the solutions have been 
        optimised with Model-Informed Variation.

        Args:
            hidden_size: int
                The size of the next hidden layer to be added.
        """
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(points_number, self.input_size),
            nn.Tanh()
        )
        pos_emb_vals = torch.empty((points_number, hidden_size))
        torch.nn.init.uniform_(pos_emb_vals, -0.01, 0.01)
        self.pos_emb = torch.nn.Parameter(pos_emb_vals)
    
    def loss(self, x: torch.Tensor, recon: torch.Tensor) -> dict:
        """
        Calculates the loss function. This is done by adding the MSE of the input and the 
        reconstruction given by the AE, as well as an L1 term multiplied by a coefficient.
        L2 is not included in this loss function as that is handled by the optimizer in 
        Pytorch.

        Args:
            x: torch.Tensor
                The input to the model of size n x W, where n is the size of the batch and W is 
                the size of each solution.
            recon: torch.Tensor
                The reconstruction of x that the model outputs.
            l1_coef: float
                The coefficient of the L1 loss term.

        Returns:
            The loss dictionary containing the total loss, reconstruction error and L1 loss.
        """
        mse = F.mse_loss(x, recon)
        #l1_loss = sum(p.abs().sum() for p in self.parameters())
        loss = mse# + l1_coef * l1_loss
        return {"loss" : loss, "recon" : mse}#, "l1" : l1_loss}
    
    def learn_from_sample(self, x: torch.Tensor, optimizer: torch.optim.Optimizer) -> dict:
        """
        Handles learning from a sample of solutions.

        Args:
            x: torch.Tensor
                The sample of solutions to learn from.
            optimizer: torch.optim.Optimizer
                The optimizer that handles the adjustment of weights.
            l1_coef: int
                The coefficient of the L1 loss term.

        Returns:
            The loss dictionary containing the total loss, reconstruction error and L1 loss. 
        """
        self.train()
        output, _ = self.forward(x)
        loss_dict = self.loss(x, output)
        loss = loss_dict["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss_dict
    
    def cosine_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor
                shape = (Nb, D)
            y: torch/Tensor
                shape = (Np, D)
        """
        x_sizes = torch.sqrt(torch.sum(torch.pow(x,2),dim=1))
        y_sizes = torch.sqrt(torch.sum(torch.pow(y,2),dim=1))
        return x.matmul(y.T) / torch.outer(x_sizes, y_sizes)