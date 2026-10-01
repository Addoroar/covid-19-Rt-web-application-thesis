import json

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
from flask import Blueprint, render_template, request, session, url_for
from scipy import stats as sps

PREFECTURES_URL = "https://raw.githubusercontent.com/Sandbird/covid19-Greece/master/prefectures.csv"
REGIONS_URL = "https://raw.githubusercontent.com/Sandbird/covid19-Greece/master/regions.csv"

R_T_MAX = 12
R_T_RANGE = np.linspace(0, R_T_MAX, R_T_MAX * 100 + 1)
GAMMA = 1 / 7

# Validated palette (dark mode) — see website/static/css/style.css for the same tokens.
SURFACE = "#1a1a19"
GRID = "#2c2c2a"
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#c3c2b7"
SERIES_1 = "#3987e5"  # blue
SERIES_3 = "#199e70"  # aqua
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_SERIOUS = "#ec835a"
STATUS_CRITICAL = "#d03b3b"

CHART_FONT = dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=TEXT_SECONDARY)


def themed_layout(**overrides):
    layout = dict(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=CHART_FONT,
        title=dict(font=dict(color=TEXT_PRIMARY, size=16)),
        legend=dict(font=dict(color=TEXT_SECONDARY)),
        margin=dict(t=48, r=24, b=48, l=56),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
    )
    for key, value in overrides.items():
        if isinstance(value, dict) and key in layout and isinstance(layout[key], dict):
            layout[key] = {**layout[key], **value}
        else:
            layout[key] = value
    return go.Layout(**layout)


plot_rt = Blueprint('plot_rt', __name__)


@plot_rt.route('/form')
def form():
    df = pd.read_csv(PREFECTURES_URL)
    states_list = df['region_en'].unique()

    return render_template("form.html", states_list=states_list)


@plot_rt.route("/plot_rt", methods=["GET", "POST"])
def plot_data():

    if request.method == "POST":
        date_from = request.form.get("dfrom")
        date_to = request.form.get("dto")
        form_state = request.form.get("fstate")
        state_name = form_state

    df = pd.read_csv(PREFECTURES_URL)

    # select the desired columns
    cols = ['date', 'region_en', 'cases']
    df_subset = df[cols]

    # group the data by region_en and date and calculate the cumulative sum of cases
    states = df_subset.groupby(['region_en', 'date'])['cases'].sum().groupby(level=[0]).cumsum()

    states_list = df['region_en'].unique()

    df = states.groupby('region_en').last().reset_index()
    df = df.sort_values(by='cases', ascending=True)

    df = df[df['cases'] > 60000]
    df = df[df['cases'] < 700000]

    x = df['region_en']
    y = df['cases']

    # Create the bar chart
    trace = go.Bar(x=x, y=y, marker=dict(color=SERIES_1))

    layout = themed_layout(
        title='Total Cases by Region above 60k',
        xaxis=dict(
            title='Region',
            tickangle=90,
            automargin=True
        ),
        yaxis=dict(
            title='Total Cases'
        ),
        bargap=0.15
    )

    fig = go.Figure(data=[trace], layout=layout)
    plot_json_bar = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    region = request.args.get('region')

    if region:
        state_name = region
        df = pd.read_csv(REGIONS_URL)

        # select the desired columns
        cols = ['date', 'region', 'cases']
        states = df[cols]

        # group the data by region and sort the result by region name
        states = states.groupby('region').apply(lambda x: x.set_index('date')['cases']).sort_index()

    def prepare_cases(cases, cutoff=5):
        new_cases = cases.diff()

        smoothed = new_cases.rolling(
            7,
            win_type='gaussian',
            min_periods=1,
            center=True
        ).mean(std=2).round()

        idx_start = np.searchsorted(smoothed, cutoff)

        smoothed = smoothed.iloc[idx_start:]
        original = new_cases.loc[smoothed.index]

        return original, smoothed

    cases = states.xs(state_name).rename(f"{state_name} cases")

    if region:
        start_date = '2021-01-01'
        end_date = '2021-02-20'
        date_from = start_date
        date_to = end_date
    else:
        start_date = date_from
        end_date = date_to
        cases = cases[start_date:end_date]

        session['start_date'] = start_date
        session['end_date'] = end_date
        session['form_state'] = form_state

    original, smoothed = prepare_cases(cases)

    x = pd.to_datetime(cases.reset_index().date)
    y_smoothed = smoothed
    y_original = original

    # RT Plot
    trace_smooth = go.Scatter(
        x=x,
        y=y_smoothed,
        mode='lines',
        name='Smoothed',
        line=dict(color=SERIES_1, width=2)
    )

    trace_original = go.Scatter(
        x=x,
        y=y_original,
        mode='lines',
        line=dict(dash='dot', color=TEXT_SECONDARY, width=1),
        name='Actual'
    )

    # Create the layout
    layout = themed_layout(
        title=f'{state_name} - New cases per day',
        xaxis=dict(title='Date', type='date'),
        yaxis=dict(title='Number of cases per day')
    )

    # Create the figure
    fig = go.Figure(data=[trace_original, trace_smooth], layout=layout)
    plot_json_smoothed = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    def get_posteriors(sr, sigma=0.15):
        # (1) Calculate Lambda
        lam = sr[:-1].values * np.exp(GAMMA * (R_T_RANGE[:, None] - 1))

        # (2) Calculate each day's likelihood
        likelihoods = pd.DataFrame(
            data=sps.poisson.pmf(sr[1:].values, lam),
            index=R_T_RANGE,
            columns=sr.index[1:])

        # (3) Create the Gaussian Matrix
        process_matrix = sps.norm(
            loc=R_T_RANGE,
            scale=sigma
        ).pdf(R_T_RANGE[:, None])

        # (3a) Normalize all rows to sum to 1
        process_matrix /= process_matrix.sum(axis=0)

        # (4) Calculate the initial prior
        prior0 = np.ones_like(R_T_RANGE) / len(R_T_RANGE)
        prior0 /= prior0.sum()

        # Create a DataFrame that will hold our posteriors for each day
        # Insert our prior as the first posterior.
        posteriors = pd.DataFrame(
            index=R_T_RANGE,
            columns=sr.index,
            data={sr.index[0]: prior0}
        )

        # (5) Iteratively apply Bayes' rule
        for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):
            # (5a) Calculate the new prior
            current_prior = process_matrix @ posteriors[previous_day]

            # (5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
            numerator = likelihoods[current_day] * current_prior

            # (5c) Calculate the denominator of Bayes' Rule P(k)
            denominator = np.sum(numerator)

            # Execute full Bayes' Rule
            posteriors[current_day] = numerator / denominator

        return posteriors

    # Note that we're fixing sigma to a value just for the example
    posteriors = get_posteriors(smoothed, sigma=.25)

    # Plot for posteriors
    fig = go.Figure()

    # Create a line plot for each column of the DataFrame
    for col in posteriors.columns:
        fig.add_trace(go.Scatter(x=posteriors.index, y=posteriors[col], name=str(col)))

    fig.update_layout(themed_layout(
        title=f'{state_name} - Posterior Distributions',
        xaxis=dict(title='Rt', range=[0, 4]),
        yaxis=dict(title='Likelihood')
    ))

    plot_json_posteriors = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    def highest_density_interval(pmf, p=.95):
        # If we pass a DataFrame, just call this recursively on the columns
        if pmf.empty:
            return pmf

        if isinstance(pmf, pd.DataFrame):
            return pd.DataFrame(
                [highest_density_interval(pmf[col], p=p) for col in pmf],
                index=pmf.columns)

        cumsum = np.cumsum(pmf.values)

        # N x N matrix of total probability mass for each low, high
        total_p = cumsum - cumsum[:, None]

        # Return all indices with total_p > p
        lows, highs = (total_p > p).nonzero()

        if len(lows) > 0:
            # Find the smallest range (highest density)
            best = (highs - lows).argmin()
            low = pmf.index[lows[best]]
            high = pmf.index[highs[best]]

            return pd.Series(
                [low, high],
                index=[f'Low_{p * 100:.0f}', f'High_{p * 100:.0f}']
            )
        else:
            return pd.Series(
                [0, 0],
                index=[f'Low_{p * 100:.0f}', f'High_{p * 100:.0f}']
            )

    # Note that this takes a while to execute - it's not the most efficient algorithm
    hdis = highest_density_interval(posteriors, p=.95)

    most_likely = posteriors.idxmax().rename('ML_Rt')

    # Concatenate the most likely Rt values for each day with the HDI into one dataframe
    result = pd.concat([most_likely, hdis], axis=1)

    x = pd.to_datetime(result.reset_index().date)
    y = result['ML_Rt']
    lower_bound = result['Low_95']
    upper_bound = result['High_95']

    colors = []
    for val in y:
        if val >= 1.5:
            colors.append(STATUS_CRITICAL)
        elif val >= 1:
            colors.append(STATUS_SERIOUS)
        elif val >= 0.5:
            colors.append(STATUS_WARNING)
        else:
            colors.append(STATUS_GOOD)

    # RT Plot
    trace = go.Scatter(
        x=x,
        y=y,
        mode='lines+markers',
        line=dict(color=TEXT_SECONDARY, width=1),
        name='Most Likely Rt',
        marker=dict(
            color=colors,
            size=8,
            line=dict(width=2, color=SURFACE)
        )
    )

    # Add upper bound data of the HDI
    u_bound = go.Scatter(
        x=x,
        y=upper_bound,
        mode='lines',
        name='Upper Bound HDI',
        line=dict(color='rgba(57, 135, 229, 0.15)', width=1),
        fillcolor='rgba(57, 135, 229, 0.12)',
        fill='tonexty'
    )

    # Add lower bound data of the HDI
    l_bound = go.Scatter(
        x=x,
        y=lower_bound,
        mode='lines',
        name='Lower Bound HDI',
        line=dict(color='rgba(57, 135, 229, 0.15)', width=1),
        fillcolor='rgba(57, 135, 229, 0.12)',
        fill='tonexty'
    )

    # Add baseline of Rt = 1 for reference
    hline = go.Scatter(
        x=[min(x), max(x)],
        y=[1, 1],
        mode='lines',
        line=dict(color=TEXT_SECONDARY, dash='dash', width=1),
        name='Rt = 1'
    )

    # Create the layout
    layout = themed_layout(
        title=f'{state_name} - Most Likely Rt per day',
        xaxis=dict(title='Date', type='date'),
        yaxis=dict(title='Rt', range=[0, max(y) + 0.5]),
        autosize=True
    )

    # Create the figure
    fig = go.Figure(data=[trace, u_bound, l_bound, hline], layout=layout)
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        'plot.html',
        plot=plot_json,
        date_from=date_from,
        date_to=date_to,
        states_list=states_list,
        css=url_for('static', filename='css/style.css'),
        html_table=result,
        plot_json_smoothed=plot_json_smoothed,
        plot_json_posteriors=plot_json_posteriors,
        plot_json_bar=plot_json_bar
    )
