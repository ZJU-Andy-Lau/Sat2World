import torch

def check_nan(input:torch.Tensor,name:str):
    if input.isnan().any():
        print(f"{name} has nan")