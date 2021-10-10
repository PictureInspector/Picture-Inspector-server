from app.main import models
from app.main import resources
from flask import Flask
from flask_restful import Api
from flask_sqlalchemy import SQLAlchemy
import os


# Create Flask application
app = Flask(__name__)


# Initialize database
dir_path = os.path.abspath(os.getcwd())
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{dir_path}/database.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# Initialize api service for the application
api = Api(app)


@app.before_first_request
def create_tables() -> None:
    """
    Initialize database tables if necessary.
    """
    db.create_all()
