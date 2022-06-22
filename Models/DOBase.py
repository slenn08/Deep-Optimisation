from typing import List
import torch
from torch import nn

from abc import abstractmethod, ABC

class DOBase(nn.Module, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x : torch.Tensor, *args, **kwargs) -> List[torch.Tensor]:
        pass

    @abstractmethod
    def encode(self, x : torch.Tensor, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, z : torch.Tensor, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def vary(self, x : torch.Tensor, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def transition(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def loss(self, x : torch.Tensor, *args, **kwargs) -> dict:
        pass

