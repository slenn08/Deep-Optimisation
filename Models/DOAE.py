import torch
import torch.nn.functional as F
from torch import nn

from Models.DOBase import DOBase

class DOAE(DOBase):
    """
    Implements the AE model in DO as defined in "Deep Optimisation: Learning and
    Searching in Deep Representations of Combinatorial Optimisation Problems", Jamie
    Caldwell.
    """
    def __init__(self, input_size: int, dropout_prob: float, device: torch.device):
        """
        Constructor method for the AE. The encoder and decoder start of as empty models, as 
        layers get added to them during subsequent transitions.

        Args:
            input_size: int
                The size of the solutions in a given combinatorial optimisation problem.
            dropout_prob: float
                The amount of dropout that occurs in the input to the model.
            device: torch.device
                The device the model is loadeded onto.
        """
        super().__init__()
        self.encoder = nn.Sequential(nn.Dropout(dropout_prob))
        self.decoder = nn.Sequential()
        self.device = device
        self.input_size = input_size   
        self.num_layers = 1
    
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Passes the input through a deterministic decoder.

        Args:
            x: torch.Tensor
                The input to be passed through the model. This is of shape n x W, where n is
                the number of solutions being passed through and W is the size of each solution.

        Returns:
            A list containing the reconstruction of the input of size n x W, and the latent point
            calculated from the input of size n x L where L is the size of the deepest latent
            space.
        """
        z = self.encoder(x)
        return [self.decoder(z), z]
    
    def encode(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        """
        Encodes the input up to a specified layer in the model.

        Args:
            x: torch.Tensor
                The input of shape n x W, where n is the number of solutions being passed
                through and W is the size of each solution.
            layer: int
                The layer number of the hidden layer used to calculate a latent representation.
        
        Returns:
            The generated latent space at the specified layer of size n x L where L is the size
            of the latent space at that layer.
        """
        return self.encoder[:1+(2*layer)](x)

    def decode(self, z: torch.Tensor, layer: int) -> torch.Tensor:
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
        z = self.encoder[1+(2*layer):](z)
        return self.decoder(z)
        #return self.decoder[(self.num_layers-layer-1)*2:](z)
    
    def encode_step(self, x: torch.Tensor, layer: int) -> torch.Tensor:
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
        s = x.clone().detach()
        h = self.encode(s, layer)

        # Determine one variable in each of the solutions to flip
        ds_indices = torch.randint(0,s.shape[1],(s.shape[0],))
        s[torch.arange(s.shape[0]), ds_indices] *= -1
        hs = self.encode(s, layer)

        dh = hs - h
        a = torch.mean(torch.abs(dh), dim=1)
        z, _ = torch.max(torch.abs(dh), dim=1)
        t = a + (z - a) * torch.rand(a.shape, device=self.device)

        # transposes are needed to ensure gt is carried out across rows
        dh = torch.where((torch.abs(dh).T > t).T, torch.sign(dh) - h, torch.zeros_like(dh))

        return dh
    
    def vary(self, x: torch.Tensor, layer: int, encode: bool) -> torch.Tensor:
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
        hidden_repr = self.encode(x, layer)
        new_hidden_repr = hidden_repr.clone().detach()

        # use standard assign method
        if not encode:
            i = torch.randint(0,hidden_repr.shape[1], (hidden_repr.shape[0],))
            # Provides values of either 1 or -1
            new_activations = torch.randint(0, 2, i.shape, dtype=torch.float32, device=self.device) * 2 - 1
            new_hidden_repr[torch.arange(hidden_repr.shape[0]), i] = new_activations
        else:
            d_h = self.encode_step(x, layer)
            new_hidden_repr += d_h

        old_reconstruction = torch.sign(self.decode(hidden_repr, layer))
        new_reconstruction = torch.sign(self.decode(new_hidden_repr, layer))
        delta_s = new_reconstruction - old_reconstruction

        new_solution = torch.sign(x + delta_s)
        return new_solution
    
    def transition(self, hidden_size : int) -> None:
        """
        Adds a new layer to the model. This should be called after the solutions have been 
        optimised with Model-Informed Variation.

        Args:
            hidden_size: int
                The size of the next hidden layer to be added.
        """
        prev_size = 0
        if not self.decoder:
            prev_size = self.input_size
        else: 
            prev_size = self.decoder[0].in_features

        weight = torch.zeros((hidden_size, prev_size))
        nn.init.uniform_(weight, -0.01, 0.01)

        encoder_layer = nn.Linear(prev_size, hidden_size, device=self.device)
        encoder_layer.weight = nn.Parameter(weight)
        decoder_layer = nn.Linear(hidden_size, prev_size)
        decoder_layer.weight = nn.Parameter(weight.transpose(0,1))

        self.encoder = nn.Sequential(*(list(self.encoder) + [encoder_layer,nn.Tanh()])).to(device=self.device)
        self.decoder = nn.Sequential(*([decoder_layer,nn.Tanh()] + list(self.decoder))).to(device=self.device)

        self.num_layers += 1
    
    def loss(self, x: torch.Tensor, recon: torch.Tensor, l1_coef: float) -> dict:
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
        l1_loss = sum(p.abs().sum() for p in self.parameters())
        loss = mse + l1_coef * l1_loss
        return {"loss" : loss, "recon" : mse, "l1" : l1_loss}
    
    def learn_from_sample(self, x: torch.Tensor, optimizer: torch.optim.Optimizer,
                          l1_coef: float) -> dict:
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
        loss_dict = self.loss(x, output, l1_coef)
        loss = loss_dict["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss_dict