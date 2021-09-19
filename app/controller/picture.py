from werkzeug.datastructures import FileStorage
from flask_restful.reqparse import RequestParser
from flask_restful import Resource
from typing import Tuple, Optional
from app.utils import save_image, retrieve_caption


class Picture(Resource):

    _IMAGE_ARG: str = 'image'

    def get(self) -> Tuple[any, int]:
        # TODO: Return caption
        caption = 'GET request has not been implemented yet :)'
        return caption, 200

    def post(self) -> Optional[Tuple[any, int]]:

        parser = RequestParser()
        parser.add_argument(
            Picture._IMAGE_ARG,
            type=FileStorage,
            location='files',
            required=True)

        args = parser.parse_args()

        image_file: FileStorage = args[Picture._IMAGE_ARG]
        image_name, image_path = save_image(file=image_file)

        caption = retrieve_caption(image_path=image_path)

        response = {
            'imageURL': image_name,
            'caption': caption
        }

        return response, 200
