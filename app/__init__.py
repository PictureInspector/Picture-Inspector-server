from app.main.logging import LOGGING_CONFIG
from app.main.database import database
from app.main.api import api
from app.main import settings
from app import entity
from app import main
from flask import Flask
import logging
import os


# Flask app initialization
app = Flask(__name__)
app.config.from_object(settings[os.environ.get('APPLICATION_ENV', 'default')])

# Logger
log = logging.getLogger('console')

# Database ORM initialization
database.init_app(app)

# Flask api initialization
api.init_app(app)
