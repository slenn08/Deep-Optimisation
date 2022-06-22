import torch

def to_int_list(x):
    try:
        x = torch.sign(x)
        x = x.tolist()
        x = [int(i) for i in x]  
    except TypeError:
        pass
    return x