import numpy as np
import torch
from PIL import Image
from app.neural_network.utils import caption_sample


device = torch.device("cpu")
vocabulary = torch.load("./app/data/dataset.pth").voc
model = torch.load("./app/data/entity.pth").to(device)


def retrieve_caption(image_path: str) -> str:

    image = Image.open(image_path)

    return caption_sample(
        sample=image,
        model=model,
        voc=vocabulary,
        device=device)
