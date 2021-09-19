import os
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from app.neural_network.models import Model
from app.neural_network.dataset import Vocabulary


def load_checkpoint(checkpoint: dict, model: Model, optimizer: optim.Optimizer) -> int:
    """
    Loads entity. optimizer and step number from the checkpoint
    :param checkpoint: Dictionary with entity state dict, optimizer state dict and step number
    :param model: Model to be loaded
    :param optimizer: Optimizer for this entity
    :return: step number
    """
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step


def save_checkpoint(checkpoint: dict, checkpoint_path: str = "checkpoint.pth") -> None:
    """
    Saves checkpoints
    :param checkpoint: Dictionary with entity state dict, optimizer state dict and step number
    :param checkpoint_path: Path where checkpoint should be saved
    :return: None
    """
    torch.save(checkpoint, checkpoint_path)


def caption_sample(sample: Image, model: Model, voc: Vocabulary, device: torch.device,
                   transform: transforms.Compose = None) -> str:
    """
    Get the caption for the given image
    :param sample: image
    :param model: entity which will produce the caption
    :param voc: vocabulary
    :param device: device on which the image will be loaded
    :param transform: transform tob be applied to the image
    :return: caption for the image
    """
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    # Apply transform and move the image to the device
    sample = transform(sample).unsqueeze(0).to(device)
    # get the caption with start token and end token removed
    return " ".join(model.caption_image(sample, voc)[1:-1])


def caption_samples(directory: str, model: Model, voc: Vocabulary, device: torch.device,
                    transform: transforms.Compose = None) -> None:
    """
    Caption several samples from one directory
    :param directory: directory where all samples are
    :param model: entity which will produce the caption
    :param voc: vocabulary
    :param device: device on which the image will be loaded
    :param transform: transform tob be applied to the image
    :return: None
    """
    # set the entity to the evaluation mode
    model.eval()
    for item in os.listdir(directory):
        # for each sample convert it to RGB
        sample = Image.open(directory + '/' + item).convert("RGB")
        # print the caption for the sample
        print(f"Caption for {item}: {caption_sample(sample, model, voc, device, transform)}")


if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    dataset = torch.load("./app/data/dataset.pth")
    model = torch.load("./app/data/entity.pth").to(device)
    caption_samples("./app/data/images", model, dataset.voc, device, dataset.transform)
