import pandas as pd
import numpy as np
import json
from flask import Flask, render_template
import plotly.graph_objs as go
import plotly

# Create the data
data = pd.Series({
    'State A': 2,
    'State B': 10,
    'State C': 3,
    'State D': 7,
    'State E': 1,
    'State F': 4,
    'State G': 6,
    'State H': 9,
    'State I': 8,
    'State J': 5
})
print(data)

# Define the color scheme
colors = ['lightgray' if x <= 5 else 'red' for x in data.values]

# Sort the data in ascending order
data_sorted = data.sort_values()

# Create the bar chart
trace = go.Bar(
    x=data_sorted.index,
    y=data_sorted.values,
    marker=dict(
        color=colors
    )
)

layout = go.Layout(
    title='Cases Per Day by State',
    xaxis=dict(
        title='State',
        tickangle=45,
        automargin=True
    ),
    yaxis=dict(
        title='Cases Per Day'
    ),
    bargap=0.1
)

fig = go.Figure(data=[trace], layout=layout)

# Convert the figure to a JSON object and render it in a template
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('plot.html', plot=json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))

if __name__ == '__main__':
    app.run(debug=True)
