import os

from flask import Flask


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev')

    from .plot_rt import plot_rt
    from .views import views

    app.register_blueprint(plot_rt, url_prefix='/')
    app.register_blueprint(views, url_prefix='/')

    return app
