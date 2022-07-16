from typing import List
import torch
from Models.DOBase import DOBase


class DOSmall(DOBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def encode(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return super().encode(x, *args, **kwargs)
    
    def decode(self, z: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return super().decode(z, *args, **kwargs)
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> List[torch.Tensor]:
        return super().forward(x, *args, **kwargs)
    
    def vary(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return super().vary(x, *args, **kwargs)
    
    def transition(self, *args, **kwargs) -> None:
        return super().transition(*args, **kwargs)
    
    def loss(self, x: torch.Tensor, *args, **kwargs) -> dict:
        return super().loss(x, *args, **kwargs)
    
    def learn_from_sample(self, samples: torch.Tensor, optimizer: torch.optim.Optimizer, *args, **kwargs) -> dict:
        return super().learn_from_sample(samples, optimizer, *args, **kwargs)