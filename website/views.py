from flask import Blueprint, render_template, current_app, send_from_directory

views = Blueprint('views', __name__)

@views.route('/')
def home():
    return render_template("home.html")


@views.route('/about')
def about():
    return render_template('about.html')


@views.route('/pdf')
def pdf():
    return send_from_directory(
        f"{current_app.static_folder}/pdf", "paper.pdf", mimetype="application/pdf"
    )