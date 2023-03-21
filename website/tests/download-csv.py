import pandas as pd

# url = "https://raw.githubusercontent.com/Sandbird/covid19-Greece/master/regions.csv"
url = "https://raw.githubusercontent.com/Sandbird/covid19-Greece/master/prefectures.csv"


# states = pd.read_csv(url,
#                      usecols=['date', 'region_en', 'cases'],
#                      parse_dates=['date'],
#                      index_col=['region_en', 'date'],
#                      squeeze=True).sort_index()

# print(states)



df = pd.read_csv(url)

# print(df[['region_en','date','cases']])

# get a list of distinct regions
regions = df['region_en'].unique()
print(regions)

# select a single region
selected_region = 'Zakynthos'

# filter the dataframe based on the selected region and sort by date
selected_df = df[df['region_en'] == selected_region].sort_values(by='date')
# print(selected_df[['region_en','date','cases']])
# print(selected_df[['region_en','date','cases']].tail(300))

selected_df = selected_df.rename(columns={'cases': 'old_cases'})
selected_df['cases'] = selected_df['old_cases'].cumsum()
selected_df = selected_df.drop(columns=['old_cases'])

# print(selected_df[['region_en','date','cases']].head(100))
selection = selected_df[['date','region_en','cases']]
# print(selection)
# print(selected_df.set_index(['date']).T)

url = 'website clone/tests/{selected_region}.csv'
selection.to_csv(url, index=False)


selected_states = selected_df.reset_index().set_index(['region_en', 'date']).sort_index()
# print(selected_states[['cases']].tail(30))