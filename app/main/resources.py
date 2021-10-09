import sqlalchemy
from werkzeug.datastructures import FileStorage
from flask_restful.reqparse import RequestParser
from app.main.models import Caption
from app.utils import save_image, retrieve_caption
from app import app

@app.route('/api/v1/pictures', methods=['POST'])
def get_caption_for_img():

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
    except sqlalchemy.exc.IntegrityError:
        return 'already saved', 200

    return 'success', 200

