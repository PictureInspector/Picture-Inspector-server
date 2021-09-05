import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from models import Model
from dataset import Vocabulary
import os


def load_checkpoint(checkpoint: dict, model: Model, optimizer: optim.Optimizer) -> int:
    """
    Loads model. optimizer and step number from the checkpoint
    :param checkpoint: Dictionary with model state dict, optimizer state dict and step number
    :param model: Model to be loaded
    :param optimizer: Optimizer for this model
    :return: step number
    """
    model.load_state_dict(checkpoint["state_dict"])
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


def caption_sample(sample: Image, model: Model, voc: Vocabulary, device: torch.device,
                   transform: transforms.Compose = None) -> str:
    """
    Get the caption for the given image
    :param sample: image
    :param model: model which will produce the caption
    :param voc: vocabulary
    :param device: device on which the image will be loaded
    :param transform: transform tob be applied to the image
    :return: caption for the image
    """
    # Apply transform and move the image to the device
    sample = transform(sample).unsqueeze(0).to(device)
    # get the caption with start token and end token removed
    return " ".join(model.caption_image(sample, voc)[1:-1])


def caption_samples(directory: str, model: Model, voc: Vocabulary, device: torch.device,
                    transform: transforms.Compose = None) -> None:
    """
    Caption several samples from one directory
    :param directory: directory where all samples are
    :param model: model which will produce the caption
    :param voc: vocabulary
    :param device: device on which the image will be loaded
    :param transform: transform tob be applied to the image
    :return: None
    """
    # set the model to the evaluation mode
    model.eval()
    for item in os.listdir(directory):
        # for each sample convert it to RGB
        sample = Image.open(directory + '/' + item).convert("RGB")
        # print the caption for the sample
        print(f"Caption for {item}: {caption_sample(sample, model, voc, device, transform)}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = torch.load("dataset.pth")
    model = torch.load("model.pth")
    caption_samples("examples", model, dataset.voc, device, dataset.transform)
