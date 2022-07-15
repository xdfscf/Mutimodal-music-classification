from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

import logging
import os
app = Flask(__name__)
app.config.from_object('config')
app.config['SQLALCHEMY_POOL_SIZE'] = 60
db = SQLAlchemy(app)

migrate = Migrate(app, db)





from app import views,models

