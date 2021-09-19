from flask_restful import Api
from app.controller.picture import Picture
from app.main.error import custom_errors


# Initialize api service
api = Api(catch_all_404s=True, errors=custom_errors, prefix='/api')

# Set endpoints
api.add_resource(Picture, '/v1/pictures')
