import torch.nn as nn
import torch.optim as optim


def load_checkpoint(checkpoint: dict, model: nn.Module, optimizer: optim.Optimizer) -> int:
    """
    Loads model. optimizer and step number from the checkpoint
    :param checkpoint: Dictionary with model state dict, optimizer state dict and step number
    :param model: Model to be loaded
    :param optimizer: Optimizer for this model
    :return: step number
    """
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step
