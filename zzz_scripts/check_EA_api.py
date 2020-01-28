import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

import datetime

# apply method
def format_ids_helper(x):
    if x != x.upper():
        x = x.upper()
    if len(x) < 6 and x.isdigit():
       zrs = 6-len(x)
       for i in range(zrs):
           x = '0' + x
    return x


# loc method       
def format_ids_loc(df, col):
    # format station ids
    for i in df.index:
        if df[col].loc[i] != df[col].loc[i].upper():
            df[col].loc[i] = df[col].loc[i].upper()
    
        if len(df[col].loc[i]) < 6 and str(df[col].loc[i])[0].isdigit():
            zrs = 6-len(df[col].loc[i])
            for i in range(zrs):
                df[col].loc[i] = '0'+df[col].loc[i]
    
    return df


# iterrows method       
def format_ids(df, col):
    # format station ids
    for index, val in df.iterrows():
        if val[col] != val[col].upper():
            df[col].loc[index] = df[col].loc[index].upper()
    
        if len(val[col]) < 6 and str(val[col])[0].isdigit():
            zrs = 6-len(val[col])
            for i in range(zrs):
                df[col].loc[index] = '0'+df[col].loc[index]
    
    return df


''' API ids'''

root = 'https://environment.data.gov.uk/flood-monitoring/id/stations?parameter=rainfall'
data = requests.get(root).json()

api_ids = pd.DataFrame(columns=['id', 'easting', 'northing'])
for i in data['items']:
    if all(['easting' in i, 'northing' in i]):
        api_ids = api_ids.append({'id': i['notation'], 'easting': i['easting'], 'northing': i['northing']}, ignore_index=True)



o = datetime.datetime.now()
api_idss = format_ids(api_ids, 'id')
print(datetime.datetime.now()-o)

o = datetime.datetime.now()
api_idszs = format_ids_loc(api_ids, 'id')
print(datetime.datetime.now()-o)

o = datetime.datetime.now()
api_ids['id'] = api_ids['id'].apply(format_ids_helper)
print(datetime.datetime.now()-o)