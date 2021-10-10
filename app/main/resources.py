from sqlalchemy.exc import IntegrityError
from werkzeug.datastructures import FileStorage
from flask_restful.reqparse import RequestParser
from app.main.models import Caption
from app.utils import save_image, retrieve_caption
from app import app


@app.route('/api/v1/pictures', methods=['POST'])
def get_caption_for_img():
    """
    This function unwraps caption post request and
    retrieves image caption using neural network.

    Request must contain multipart form data
    with key 'image' for target image.

    :return: Response structure {'imageURL': str, 'caption': str}
             and the http status code.
    """

    parser = RequestParser()
    parser.add_argument(
        'image',
        type=FileStorage,
        location='files',
        required=True
    )
    
    args = parser.parse_args()
    
    image_file: FileStorage = args['image']
    image_name, image_path = save_image(file=image_file)
    
    caption = retrieve_caption(image_path=image_path)[:-2]
    
    response = {
        'imageURL': image_name,
        'caption': caption
    }
    
    return response, 200


@app.route('/api/v1/feedback', methods=['POST'])
def save_feedback():
    """
    This function unwraps feedback post request and saves the feedback.
    Request must contain json structure {'image_url': str, 'is_good': int}
    where is_good is translated to bool as int(is_good).
    After that, the request is saved to the database.

    :return: Status message and http status code.
    """

    parser = RequestParser()
    
    parser.add_argument(
        'image_url',
        type=str,
        required=True
    )

    parser.add_argument(
        'is_good',
        type=int,
        required=True
    )
    
    args = parser.parse_args()
    
    caption_model = Caption(
        image_url=args['image_url'],
        isGood=bool(args['is_good'])
    )

    try:
        caption_model.add()
    except IntegrityError:
        return 'already saved', 200

    return 'success', 200

