from sqlalchemy import Integer, String
from app import database


class Caption(database.Model):

    __tablename__ = "caption"

    id = database.Column(Integer, primary_key=True)
    author_id = database.Column(Integer, nullable=False)
    image_url = database.Column(String, unique=True, nullable=False)
    caption = database.Column(String(500))

    def __repr__(self):
        return (f'<Caption(id: {self.id}, imageURL: {self.image_url}'
                f', author: {self.author_id}, caption: {self.caption})>')
