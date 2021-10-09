import numpy as np
import torch
from PIL import Image
from app.neural_network.utils import caption_sample


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocabulary = torch.load("./app/data/dataset.pth", map_location=device).voc
model = torch.load("./app/data/entity.pth", map_location=device).to(device)
model.eval()


def retrieve_caption(image_path: str) -> str:
    print(image_path)

    image = Image.open(image_path)

    return caption_sample(
        sample=image,
        model=model,
        voc=vocabulary,
        device=device)
