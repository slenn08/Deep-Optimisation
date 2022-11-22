from abc import abstractmethod, ABC

import torch
from torch import nn

class DOBase(nn.Module, ABC):
    """
    Superclass used to define a model used for Deep Optimisation.

    Outlines several methods that such a model needs to have, such as encoding, decoding,
    varying solutions, and transitioning.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> list[torch.Tensor]:
        """
        Passes an input through the encoder and also decodes it into a reconstruction
        of the input.

        Args:
            x: torch.Tensor
                The input to be reconstructed.
        
        Returns:
            A list of tensors, where each one relates to a relevant value outputted by
            the model (the end reconstruction should be included in this).
        """
        pass

    @abstractmethod
    def encode(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Encodes the input into the latent space.

        Args:
            x: torch.Tensor
                The input that lies in the solution space.
        
        Returns:
            The latent representation of the input.
        """
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Decodes a point in the latent space back into the solution space.

        Args:
            z: torch.Tensor
                The latent points to be decoded.
        
        Returns:
            The reconstruction of the input related to the latent points.
        """
        pass

    @abstractmethod
    def vary(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Outlines the method used to vary a solution with Model-Informed Variation.

        Args:
            x: torch.Tensor
                The solutions to be inputted and subsequently varied by the model.
            
        Returns:
            New solutions after variation has been applied to them.
        """
        pass

    @abstractmethod
    def transition(self, *args, **kwargs) -> None:
        """
        Applies a transition to the model. This involves changing the model in such a way 
        that it is better able to encode solutions.
        """
        pass

    @abstractmethod
    def loss(self, x: torch.Tensor, *args, **kwargs) -> dict:
        """
        Calculated the loss of the model given an input.

        Args:
            x: torch.Tensor
                The input that the loss will be calculated w.r.t
        
        Returns:
            A dictionary containing the relevant parts of the loss function (e.g. l1 and l2,
            reconstruction loss, etc).
        """
        pass

    @abstractmethod
    def learn_from_sample(self, samples: torch.Tensor, optimizer: torch.optim.Optimizer,
                          *args, **kwargs) -> dict:
        """
        Make the model learn from a sample of the solutions.

        Args:
            samples: torch.Tensor
                The samples to learn from.
            optimizer: torch.optim.Optimizer
                The optimizer that will adjust the weights of the model
        
        Returns:
            The loss dict specifying the different parts of the loss function.
        """
        pass

