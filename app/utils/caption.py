import numpy as np
import torch
from PIL import Image
from app.neural_network.utils import caption_sample


model = np.NAN
vocabulary = np.NAN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def retrieve_caption(image_path: str) -> str:

    image = Image.open(image_path)

    return caption_sample(
        sample=image,
        model=model,
        voc=vocabulary,
        device=device)
