from PIL import Image
from app.neural_network.utils import caption_sample


def retrieve_caption(image_path: str) -> str:

    image = Image.open(image_path)

    return caption_sample(sample=image, model=None, voc=None, device=None)
