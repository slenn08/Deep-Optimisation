from typing import List
import torch
import torch.nn.functional as F

from torch import nn
import random

from DOBase import DOBase

class DO(DOBase):
    def __init__(self, input_size : int, dropout_prob : float):
        super().__init__()

        self.encoder = nn.Sequential(nn.Dropout(dropout_prob))
        self.decoder = nn.Sequential()
        self.input_size = input_size   
        self.num_layers = 1
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        z = self.encoder(x)
        return [self.decoder(z), z]
    
    def encode(self, x: torch.Tensor, layer : int) -> torch.Tensor:
        return self.encoder[:1+(2*layer)](x)

    def decode(self, z: torch.Tensor, layer : int) -> torch.Tensor:
        return self.decoder[(self.num_layers-layer-1)*2:](z)
    
    def encode_step(self, x : torch.Tensor, layer : int) -> torch.Tensor:
        s = x.clone().detach()
        h = self.encode(s, layer)

        ds_indices = torch.randint(0,s.shape[1],(s.shape[0],))
        s[torch.arange(s.shape[0]), ds_indices] *= -1
        hs = self.encode(s, layer)

        dh = hs - h
        a = torch.mean(torch.abs(dh), dim=1)
        z = torch.max(torch.abs(dh), dim=1)
        t = a + (z - a) * torch.rand(a.shape)

        # transposes are needed to ensure gt is carried out across rows
        dh = torch.where((torch.abs(dh).T > t).T, torch.sign(dh) - h, torch.zeros_like(dh))

        return dh
    
    def vary(self, x: torch.Tensor, layer : int, encode : bool) -> torch.Tensor:
        hidden_repr = self.encode(x, layer)
        new_hidden_repr = hidden_repr.clone().detach()

        # hillclimb or use standard assign method
        if not encode or layer == 0:
            i = torch.randint(0,hidden_repr.shape[1], (hidden_repr.shape[0],))
            # Provides values of either 1 or -1
            new_activations = torch.randint(0,2,i.shape,dtype=torch.float32) * 2 - 1
            new_hidden_repr[torch.arange(hidden_repr.shape[0]),i] = new_activations
        else:
            d_h = self.encode_step(x, layer)
            new_hidden_repr += d_h

        old_reconstruction = torch.sign(self.decode(hidden_repr, layer))
        new_reconstruction = torch.sign(self.decode(new_hidden_repr, layer))
        delta_s = new_reconstruction - old_reconstruction

        new_solution = x + delta_s
        return new_solution
    
    def transition(self, hidden_size : int) -> None:
        prev_size = 0
        if not self.decoder:
            prev_size = self.input_size
        else: 
            prev_size = self.decoder[0].in_features

        weight = torch.tensor([[random.uniform(-0.01,0.01) for _ in range(prev_size)]
                                         for _ in range(hidden_size)], requires_grad=True)
        encoder_layer = nn.Linear(prev_size, hidden_size)
        encoder_layer.weight = nn.Parameter(weight)
        decoder_layer = nn.Linear(hidden_size, prev_size)
        decoder_layer.weight = nn.Parameter(weight.transpose(0,1))

        self.encoder = nn.Sequential(*(list(self.encoder) + [encoder_layer,nn.Tanh()]))
        self.decoder = nn.Sequential(*([decoder_layer,nn.Tanh()] + list(self.decoder)))

        self.num_layers += 1
    
    def loss(self, x : torch.Tensor, recon : torch.Tensor,
             l1_coef : float) -> dict:
        mse = F.mse_loss(x, recon)
        l1_loss = sum(p.abs().sum() for p in self.float.parameters())
        loss = mse + l1_coef * l1_loss
        return {"loss" : loss.item(), "recon" : mse, "l1" : l1_loss}