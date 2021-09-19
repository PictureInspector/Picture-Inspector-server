import torch
from PIL import Image
from app.neural_network.utils import caption_sample


model = torch.load('./app/neural_network/model.pth')
vocabulary = torch.load('./app/neural_network/dataset.pth').voc
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def retrieve_caption(image_path: str) -> str:

    image = Image.open(image_path)

    return caption_sample(
        sample=image,
        model=model,
        voc=vocabulary,
        device=device)
