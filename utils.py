import torch
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


def save_checkpoint(checkpoint: dict, checkpoint_path: str = "checkpoint.pth") -> None:
    """
    Saves checkpoints
    :param checkpoint: Dictionary with model state dict, optimizer state dict and step number
    :param checkpoint_path: Path where checkpoint should be saved
    :return: None
    """
    torch.save(checkpoint, checkpoint_path)
