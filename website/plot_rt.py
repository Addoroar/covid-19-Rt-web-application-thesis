import json

import numpy as np
import pandas as pd
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

    # Plain data payload for the Chart.js-rendered bar chart (see plot.html)
    bar_chart_data = {
        'labels': df['region_en'].tolist(),
        'cases': df['cases'].tolist(),
        'color': SERIES_1,
        'title': 'Total Cases by Region above 60k',
    }
    bar_chart_json = json.dumps(bar_chart_data)

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

    # Plain data payload for the Chart.js-rendered cases chart (see plot.html)
    cases_chart_data = {
        'labels': [d.strftime('%Y-%m-%d') for d in x],
        'actual': [None if pd.isna(v) else round(v, 2) for v in y_original],
        'smoothed': [None if pd.isna(v) else round(v, 2) for v in y_smoothed],
        'color': SERIES_1,
        'title': f'{state_name} - New cases per day',
    }
    cases_chart_json = json.dumps(cases_chart_data)

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

    # Plain data payload for the Chart.js-rendered posteriors chart (see plot.html)
    # Restrict to Rt in [0, 4] (matches the original axis range) and downsample
    # the 1201-point R_T_RANGE so each of the per-day lines stays cheap to draw.
    posteriors_view = posteriors.loc[(posteriors.index >= 0) & (posteriors.index <= 4)].iloc[::4]
    posteriors_chart_data = {
        'labels': [round(v, 2) for v in posteriors_view.index],
        'series': [
            {'name': str(col), 'data': [round(v, 5) for v in posteriors_view[col]]}
            for col in posteriors_view.columns
        ],
        'title': f'{state_name} - Posterior Distributions',
    }
    posteriors_chart_json = json.dumps(posteriors_chart_data)

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

    y_axis_max = max(y.max(), upper_bound.replace(0, np.nan).quantile(0.95)) + 0.5

    # Plain data payload for the Chart.js-rendered Rt chart (see plot.html)
    rt_chart_data = {
        'labels': [d.strftime('%Y-%m-%d') for d in x],
        'ml_rt': [round(v, 2) for v in y],
        'low_95': [round(v, 2) for v in lower_bound],
        'high_95': [round(v, 2) for v in upper_bound],
        'colors': colors,
        'y_max': round(y_axis_max, 2),
        'title': f'{state_name} - Most Likely Rt per day',
    }
    rt_chart_json = json.dumps(rt_chart_data)

    return render_template(
        'plot.html',
        rt_chart_data=rt_chart_json,
        date_from=date_from,
        date_to=date_to,
        states_list=states_list,
        css=url_for('static', filename='css/style.css'),
        html_table=result,
        cases_chart_data=cases_chart_json,
        posteriors_chart_data=posteriors_chart_json,
        bar_chart_data=bar_chart_json
    )
