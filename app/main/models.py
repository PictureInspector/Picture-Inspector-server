from sqlalchemy import Integer, String, Boolean
from app import db


class Caption(db.Model):
    """
    This model represents image caption feedback.
    Each entry contains a unique image url, and the boolean feedback value.
    """

    __tablename__ = "caption"

    id = db.Column(Integer, primary_key=True)
    image_url = db.Column(String(512), unique=True)
    isGood = db.Column(Boolean)

    def add(self):
        db.session.add(self)
        db.session.commit()
