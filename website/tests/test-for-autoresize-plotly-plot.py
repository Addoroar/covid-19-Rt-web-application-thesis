import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd

# create some example data
x = [1, 2, 3, 4, 5]
y = [1, 4, 2, 3, 5]

# create a Plotly figure
fig = go.Figure(data=go.Scatter(x=x, y=y))

# save the figure as an HTML file
pio.write_html(fig, file='plot.html', auto_open=True)

