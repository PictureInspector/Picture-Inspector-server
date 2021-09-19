import os


class Config:

    BASE_DIR = os.path.join(os.pardir, os.path.dirname(__file__))
    SECRET_KEY = os.environ.get("SECRET_KEY")

    # Flask
    DEBUG = False
    TESTING = False
    PORT = 8000

    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    SQLALCHEMY_TRACK_MODIFICATIONS = True
    SQLALCHEMY_ENGINE_OPTIONS = {
        'executemany_mode': 'batch',
        'client_encoding': 'utf8',
        'case_sensitive': False,
        'echo': True,
        'echo_pool': True
    }


class DevelopmentConfig(Config):

    ENV = os.environ.get("FLASK_ENV", "development")
    DEBUG = True
    ASSETS_DEBUG = True


class TestingConfig(Config):

    ENV = os.environ.get("FLASK_ENV", "testing")
    DEBUG = True
    TESTING = True


class ProductionConfig(Config):

    ENV = os.environ.get("FLASK_ENV", "production")
    DEBUG = False
    USE_RELOADER = False


settings = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig,
}
