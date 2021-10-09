# from app.main.logging import LOGGING_CONFIG
# from app.main.database import database
# from app.main.api import api
# from app.main import settings
# from app import entity
# from app import main
# from flask import Flask
# import logging
# import os

# from flask_sqlalchemy import SQLAlchemy

# # Flask app initialization
# app = Flask(__name__)
# app.config.from_object(settings[os.environ.get('APPLICATION_ENV', 'default')])

# basedir = os.path.abspath(os.path.dirname(__name__))
# app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(basedir, '/feedback.db') }"

# # Logger
# log = logging.getLogger('console')

# # Database ORM initialization
# database = SQLAlchemy(app)
# database.init_app(app)

# # Flask api initialization
# api.init_app(app)

import string
import random
from flask import Flask
from flask_restful import Api
from flask_sqlalchemy import SQLAlchemy
import os


app = Flask(__name__)

file_path = os.path.abspath(os.getcwd())+"/database.db"
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///"+file_path
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

from app.main import models
from app.main import resources

api = Api(app)

app.secret_key = ''.join(random.choice(string.printable) for i in range(16))
app.config['SESSION_TYPE'] = 'filesystem'


db.create_all()
# @app.before_first_request
# def create_tables():    

