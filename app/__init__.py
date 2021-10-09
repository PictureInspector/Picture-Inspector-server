from flask import Flask
from flask_restful import Api
from flask_sqlalchemy import SQLAlchemy
import os


app = Flask(__name__)

dir_path = os.path.abspath(os.getcwd())
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{dir_path}/database.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

from app.main import models
from app.main import resources

api = Api(app)

@app.before_first_request
def create_tables():
  db.create_all()    

