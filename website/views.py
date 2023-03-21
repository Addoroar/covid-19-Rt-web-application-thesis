from flask import Blueprint, render_template, Response

views = Blueprint('views', __name__)

@views.route('/')
def home():
    return render_template("home.html")


@views.route('/about')
def about():
    return render_template('about.html')
    
@views.route('/pdf')
def pdf():
    with open("D:/XAMPP/htdocs/website/flask_test/website/static/pdf/paper.pdf", 'rb') as f:
        pdf = f.read()
    return Response(pdf, content_type='application/pdf')