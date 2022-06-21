import torch
from torch import nn

from abc import abstractmethod, ABC

class DOBase(nn.Module, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x : torch.tensor) -> torch.tensor:
        pass

    @abstractmethod
    def encode(self, x : torch.tensor, *args, **kwargs) -> torch.tensor:
        pass

    @abstractmethod
    def decode(self, z : torch.tensor, *args, **kwargs) -> torch.tensor:
        pass

    @abstractmethod
    def step(self, z : torch.tensor, *args, **kwargs) -> torch.tensor:
        pass

    @abstractmethod
    def transition(self) -> None:
        pass
    
    @abstractmethod
    def vary(self, x : torch.tensor, *args, **kwargs) -> torch.tensor:
        pass