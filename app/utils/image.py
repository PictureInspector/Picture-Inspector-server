from werkzeug.datastructures import FileStorage
from typing import Tuple
from uuid import uuid4
import os


# Folder that contains user images.
IMAGE_FOLDER = os.path.join('.', 'app', 'data', 'images')


def save_image(file: FileStorage) -> Tuple[str, str]:
    """
    Save image FileStorage file to the folder './app/data/images/'.

    :param file: Image file as FileStorage.

    :return: Saved image name and image path.
    """

    image_name = str(uuid4()) + os.path.splitext(file.filename)[1]
    image_path = os.path.join(IMAGE_FOLDER, image_name)

    if not os.path.exists(image_path):
        os.makedirs(IMAGE_FOLDER, exist_ok=True)

    file.save(os.path.join(IMAGE_FOLDER, image_name))

    return image_name, image_path
