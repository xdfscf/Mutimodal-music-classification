WTF_CSRF_ENABLED = True
import os

# You should change this



basedir = os.path.abspath(os.path.dirname(__file__))


SECRET_KEY = 'hard to guess'
SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:Q13579qscesz@127.0.0.1:3306/musicdb'
SQLALCHEMY_MIGRATE_REPO = os.path.join(basedir, 'db_repository')
SQLALCHEMY_TRACK_MODIFICATIONS = True
SQLALCHEMY_COMMIT_ON_TEARDOWN = True
