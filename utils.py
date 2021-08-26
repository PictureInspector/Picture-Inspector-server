import torch


def adjust_lr(optimizer: torch.optim.Optimizer, factor: float) -> None:
    """
    Decay the learning rate by the specified factor
    :param optimizer: optimizer which learning rate should be decayed
    :param factor: factor which the learning rate should be multiplied by
    :return: None
    """
    print("\n Decaying learning rate")
    for param_group in optimizer.param_groups:
        param_group['lr'] *= factor
    print(f"The new learning rate is {optimizer.param_groups[0]['lr']}")
