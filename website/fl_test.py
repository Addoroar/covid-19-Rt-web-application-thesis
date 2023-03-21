# importing Flask and other modules
from flask import Flask, request, render_template
 
# Flask constructor
app = Flask(__name__)  
 
# A decorator used to tell the application
# which URL is associated function
@app.route('/', methods =["GET", "POST"])
def gfg():
    if request.method == "POST":
       # getting input with name = dfrom in HTML form
       date_from = request.form.get("dfrom")
       print(date_from)
       # getting input with name = dto in HTML form
       date_to = request.form.get("dto")
       print(date_to)
       return "Period: "+ date_from + " " + date_to
    return render_template("form.html")
 
if __name__=='__main__':
   app.run()