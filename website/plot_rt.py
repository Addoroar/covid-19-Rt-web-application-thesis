from flask import Flask, Response, render_template, request, redirect, url_for, Blueprint, session
import plotly.graph_objs as go
import plotly
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np



import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from scipy import stats as sps
from scipy.interpolate import interp1d

from IPython.display import clear_output



# app = Flask(__name__)

plot_rt = Blueprint('plot_rt', __name__)

@plot_rt.route('/form')
def form():   
    
    url = "https://raw.githubusercontent.com/Sandbird/covid19-Greece/master/prefectures.csv"    
    
    # read the CSV file into a DataFrame
    df = pd.read_csv(url)

    states_list = df['region_en'].unique()
    
    
    return render_template("form.html",states_list=states_list)




@plot_rt.route("/plot_rt",methods=["GET","POST"])
def plot_data():
    
    
    # # get data from request
    # data = request.get_json()

    # # check if data is empty
    # if not data:
    #     return jsonify({'error': 'empty data'})

    # # convert data to pandas dataframe
    # df = pd.DataFrame(data)

    # # check if df is empty
    # if df.empty:
    #     return jsonify({'error': 'empty dataframe'})
    
    
    
    
    
    
    if request.method == "POST":
       date_from = request.form.get("dfrom")
       date_to = request.form.get("dto")
       form_state = request.form.get("fstate")
       state_name = form_state
    
     
       
    url = "https://raw.githubusercontent.com/Sandbird/covid19-Greece/master/prefectures.csv"

    # read the CSV file into a DataFrame
    df = pd.read_csv(url)

    # select the desired columns
    cols = ['date', 'region_en', 'cases']
    df_subset = df[cols]

    # group the data by region_en and date and calculate the cumulative sum of cases
    states = df_subset.groupby(['region_en', 'date'])['cases'].sum().groupby(level=[0]).cumsum()
    
    states_list = df['region_en'].unique()
    
    
    
    
    
    
    
    



    # print(states_sorted.info())
    df = states.groupby('region_en').last().reset_index()
    df = df.sort_values(by='cases', ascending=True)
    # print(df)
    
    df = df[df['cases']>60000]
    df = df[df['cases']<700000]
    # print(df)
    
    x = df['region_en']
    y = df['cases']

    # # Define the color scheme
    # colors = ['lightgray' if x <= 5 else 'red' for x in states.values]

    # Create the bar chart
    trace = go.Bar(
        x=x,
        y=y,
        # marker=dict(
        #     color=colors
        # )
    )

    layout = go.Layout(
        title='Total Cases by Region above 60k',
        xaxis=dict(
            title='Region',
            tickangle=90,
            automargin=True
        ),
        yaxis=dict(
            title='Total Cases'
        ),
        bargap=0.1
    )

    fig = go.Figure(data=[trace], layout=layout)
    
    # Convert the figure to JSON
    plot_json_bar = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    
    
    
    
    
    
    
    region = request.args.get('region')
    print(region)
    
    if region:
        state_name=region
        url = "https://raw.githubusercontent.com/Sandbird/covid19-Greece/master/regions.csv"
        df = pd.read_csv(url)
        # print(df, df.info())

        # select the desired columns
        cols = ['date', 'region', 'cases']
        states = df[cols]
        
        # group the data by region and sort the result by region name
        states = states.groupby('region').apply(lambda x: x.set_index('date')['cases']).sort_index()
        
        # select the cases column and convert it to a Series
        # states = pd.Series(states['cases'].values, index=states['date'])

        # # group the data by region and calculate the sum of cases
        # states = states.groupby(['region','date']).groupby(level=[0])
    
    print(states)
    
    
    
    
    
    
    
    
    
    

    def prepare_cases(cases, cutoff=5):
        new_cases = cases.diff()

        smoothed = new_cases.rolling(7,
            win_type='gaussian',
            min_periods=1,
            center=True).mean(std=2).round()
        
        idx_start = np.searchsorted(smoothed, cutoff)
        # print(idx_start)
        
        # idx_start = 1
        
        smoothed = smoothed.iloc[idx_start:]
        original = new_cases.loc[smoothed.index]
        
        return original, smoothed

    cases = states.xs(state_name).rename(f"{state_name} cases")
    # print(cases.info())
    # print(cases[200:320])
    
    # cases['date']=pd.to_datetime(cases['date'])
    # start_date = '2021-01-01'
    # end_date = '2021-04-25'
    # filtered_cases = cases[start_date:end_date]
    # print(filtered_cases)

    
    if region:
        start_date = '2021-01-01'
        end_date = '2021-02-20'
        date_from = start_date
        date_to = end_date
    else:
        start_date = date_from
        end_date = date_to
        cases = cases[start_date:end_date]
        # print(cases)
        
        session['start_date'] = start_date
        session['end_date'] = end_date
        session['form_state'] = form_state
    
    
    
    # # convert series to dataframe
    # cases = cases.to_frame().reset_index()

    # # filter by date range
    # start_date = date_from
    # end_date = date_to
    # mask = (cases['date'] >= start_date) & (cases['date'] <= end_date)
    # cases = cases.loc[mask]

    # # set index
    # cases = cases.set_index('date')

    # # rename columns
    # cases.columns = [f"{state_name} cases"]

    
    
    
    # cases = cases.set_index('date').loc[start_date:end_date]
    # print(cases.shape)

    
    # print(cases[['date','East Attica cases']])
    # print(cases['East Attica cases'].dtype)

    original, smoothed = prepare_cases(cases)
    # print(smoothed)
    
    
    
    
    x = cases.reset_index().date
    y_smoothed = smoothed
    y_original = original
    
    
    # RT Plot
    trace_smooth = go.Scatter(
        x=x,
        y=y_smoothed,
        mode='lines',
        name='Smoothed',
        line=dict(color='blue')
    )
    
    trace_original = go.Scatter(
        x=x,
        y=y_original,
        mode='lines',
        line=dict(dash='dot', color='gray', width=1),
        name='Actual'
    )
    
    # Create the layout
    layout = go.Layout(
        title=f'{state_name} - New cases per day',
        xaxis=dict(title='Date', type='date'),
        yaxis=dict(title='Number of cases per day')
    )

    # Create the figure
    fig = go.Figure(data=[trace_original, trace_smooth], layout=layout)
    
    # Convert the figure to JSON
    plot_json_smoothed = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    R_T_MAX = 12
    r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)

    GAMMA=1/7
    
    
    
    def get_posteriors(sr, sigma=0.15):

        # Check if sr is empty
        # if sr.empty:
        #     return pd.Series()
        
        # (1) Calculate Lambda
        lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))

        # (2) Calculate each day's likelihood
        likelihoods = pd.DataFrame(
            data = sps.poisson.pmf(sr[1:].values, lam),
            index = r_t_range,
            columns = sr.index[1:])
        
        # (3) Create the Gaussian Matrix
        process_matrix = sps.norm(loc=r_t_range,
                                scale=sigma
                                ).pdf(r_t_range[:, None]) 

        # (3a) Normalize all rows to sum to 1
        process_matrix /= process_matrix.sum(axis=0)
        
        # (4) Calculate the initial prior
        #prior0 = sps.gamma(a=4).pdf(r_t_range)
        prior0 = np.ones_like(r_t_range)/len(r_t_range)
        prior0 /= prior0.sum()

        # Create a DataFrame that will hold our posteriors for each day
        # Insert our prior as the first posterior.
        posteriors = pd.DataFrame(
            index=r_t_range,
            columns=sr.index,
            data={sr.index[0]: prior0}
        )

        # (5) Iteratively apply Bayes' rule
        for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):

            #(5a) Calculate the new prior
            current_prior = process_matrix @ posteriors[previous_day]
            
            #(5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
            numerator = likelihoods[current_day] * current_prior
            
            #(5c) Calcluate the denominator of Bayes' Rule P(k)
            denominator = np.sum(numerator)
            
            # Execute full Bayes' Rule
            posteriors[current_day] = numerator/denominator
        
        return posteriors

    # Note that we're fixing sigma to a value just for the example
    posteriors= get_posteriors(smoothed, sigma=.25)
    # print(posteriors.index)
    # print(posteriors.columns)
    
    
    # Plot for posteriors
    # Initialize the figure
    fig = go.Figure()
    
    # Create a line plot for each column of the DataFrame
    for col in posteriors.columns:
        fig.add_trace(go.Scatter(x=posteriors.index, y=posteriors[col], name=str(col)))

    # Update the plot layout
    fig.update_layout(
        title=f'{state_name} - Posterior Distributions',
        xaxis_title='Rt',
        yaxis_title='Likelihood',
        xaxis=dict(range=[0, 4])
        )
    
    plot_json_posteriors = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    
    
    
    
    # # HDI Function
    # def highest_density_interval(pmf, p=.95, debug=True):
        
    #     # Check if pmf is empty
    #     if pmf.empty:
    #         return pmf
        
    #     # If we pass a DataFrame, just call this recursively on the columns
    #     if(isinstance(pmf, pd.DataFrame)):
    #         return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],
    #                             index=pmf.columns)
        
    #     cumsum = np.cumsum(pmf.values)
        
    #     # N x N matrix of total probability mass for each low, high
    #     total_p = cumsum - cumsum[:, None]
        
    #     # Return all indices with total_p > p
    #     lows, highs = (total_p > p).nonzero()
        
    #     # Find the smallest range (highest density)
    #     # best = (highs - lows).argmin()
        
    #     if (highs - lows).all() > 0:
    #         best = (highs - lows).argmin()
    #     else:
    #         print('array is empty')
        
    #     low = pmf.index[lows[best]]
    #     high = pmf.index[highs[best]]
        
    #     return pd.Series([low, high],
    #                     index=[f'Low_{p*100:.0f}',
    #                             f'High_{p*100:.0f}'])

    # hdi = highest_density_interval(posteriors, debug=True)
    # # hdi.tail()
    
    
    
    
    
    
    # HDI Function
    def highest_density_interval(pmf, p=.95, debug=False):
        
        # Check if pmf is empty
        if pmf.empty:
            return pmf
        
        # If we pass a DataFrame, just call this recursively on the columns
        if(isinstance(pmf, pd.DataFrame)):
            return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],
                                index=pmf.columns)
        
        cumsum = np.cumsum(pmf.values)
        
        # N x N matrix of total probability mass for each low, high
        total_p = cumsum - cumsum[:, None]
        
        # Return all indices with total_p > p
        lows, highs = (total_p > p).nonzero()
        
        # Find the smallest range (highest density)
        if len(lows)>0:
            best = (highs - lows).argmin()
            low = pmf.index[lows[best]]
            high = pmf.index[highs[best]]
        
        
            return pd.Series([low, high],
                             index=[f'Low_{p*100:.0f}',
                                    f'High_{p*100:.0f}'])
        else:
            return pd.Series([0, 0],
                             index=[f'Low_{p*100:.0f}',
                                    f'High_{p*100:.0f}'])

    hdi = highest_density_interval(posteriors, debug=True)
    # hdi.tail()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Note that this takes a while to execute - it's not the most efficient algorithm
    hdis = highest_density_interval(posteriors, p=.95)

    most_likely = posteriors.idxmax().rename('ML_Rt')

    # Concatenate the most likely Rt values for each day with the HDI into one dataframe
    result = pd.concat([most_likely, hdis], axis=1)
    # html_table = result.to_html(index=False)
    # result.tail()
    
    
    
    
    
    
    
    x = result.reset_index().date
    y = result['ML_Rt']
    lower_bound = result['Low_95']
    upper_bound = result['High_95']
    
    colors = []

    for val in y:
        if val >= 1.5:
            colors.append('red')
        elif val >= 1:
            colors.append('salmon')
        elif val >= 0.5:
            colors.append('gray')
        else:
            colors.append('black')


    # RT Plot
    trace = go.Scatter(
        x=x,
        y=y,
        mode='lines+markers',
        line=dict(color='black', width=1),
        name='Most Likely Rt',
        marker=dict(
            color=colors,
            size=8,
            line=dict(width=1, color='black')
        )
    )
    
    # confidence_interval = go.Scatter(
    #     x=np.concatenate([x, x[::-1]]), 
    #     y=np.concatenate([upper_bound, lower_bound[::-1]]),
    #     fill='tonexty', 
    #     fillcolor='rgba(0, 0, 255, 0.2)',
    #     line=dict(color='rgba(0, 0, 255, 0.2)', width=1),
    #     name='HDI'
    # )
    
    # Add upper bound data of the HDI
    u_bound = go.Scatter(
        x=x,
        y=upper_bound,
        mode='lines',
        name='Upper Bound HDI',
        line=dict(color='rgba(0, 0, 255, 0.1)', width=1),
        fillcolor='rgba(0, 0, 255, 0.1)',
        fill='tonexty'
    )
    
    # Add lower bound data of the HDI
    l_bound = go.Scatter(
        x=x,
        y=lower_bound,
        mode='lines',
        name='Lower Bound HDI',
        line=dict(color='rgba(0, 0, 255, 0.1)', width=1),
        fillcolor='rgba(0, 0, 255, 0.1)',
        fill='tonexty'
    )
    
    #Add baseline of Rt = 1 for reference
    hline = go.Scatter(
        x=[min(x), max(x)],
        y=[1, 1],
        mode='lines',
        line=dict(color='black', dash='dash', width=1),
        name='Rt = 1'
    )
    
    # Create the layout
    layout = go.Layout(
        title=f'{state_name} - Most Likely Rt per day',
        xaxis=dict(title='Date', type='date'),
        yaxis=dict(title='Rt', range=[0, max(y)+0.5]),
        autosize=True
    )

    # Create the figure
    fig = go.Figure(data=[trace, u_bound, l_bound, hline], layout=layout)
    
    # Convert the figure to JSON
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)    

    # Render the template with the JSON data
    return render_template('plot.html', plot=plot_json, date_from=date_from, date_to=date_to, states_list=states_list, css=url_for('static', filename='css/style.css'), html_table=result, plot_json_smoothed=plot_json_smoothed, plot_json_posteriors=plot_json_posteriors, plot_json_bar=plot_json_bar)


# if __name__ == '__main__':
#     plot_rt.run(debug=True)










