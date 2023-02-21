from flask import Flask
from os import path


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'

    from .plot_rt import plot_rt
    from .views import views

    app.register_blueprint(plot_rt, url_prefix='/')
    app.register_blueprint(views, url_prefix='/')

    return app
