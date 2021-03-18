import torch
from config import config

def get_device():
    return torch.device(config["device"])

def zeros(*args):
    return torch.zeros(*args, device=get_device())

def ones(*args):
    return torch.ones(*args, device=get_device())
