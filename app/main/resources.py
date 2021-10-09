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
    
    caption = retrieve_caption(image_path=image_path)
    
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
        'caption',
        type=str,
        required=True
    )

    parser.add_argument(
        'is_good',
        type=bool,
        required=True
    )
    
    args = parser.parse_args()
    
    caption_model = Caption(
        image_url=args['image_url'],
        caption=args['caption'],
        isGood=args['is_good']
    )

    caption_model.add()

    return 'feedback saved successfully', 200

