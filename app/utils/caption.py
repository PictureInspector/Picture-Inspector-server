import torch
from PIL import Image
from app.neural_network.utils import caption_sample


# Load model neural network model to produce image captions.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocabulary = torch.load("./app/data/dataset.pth", map_location=device).voc
model = torch.load("./app/data/entity.pth", map_location=device).to(device)
model.eval()


def retrieve_caption(image_path: str) -> str:
    """
    Generate image caption for image with specified path.

    :param image_path: Path of the existing image file.

    :return: Generated image caption.
    """
    
    image = Image.open(image_path)

    return caption_sample(
        sample=image,
        model=model,
        voc=vocabulary,
        device=device)
