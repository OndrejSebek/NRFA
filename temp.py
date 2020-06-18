import pandas as pd

m = pd.read_excel('meta/EA_gdflive_newnew.xlsx', dtype=str)

n = pd.read_csv('meta/list.csv', header=None, dtype=str)
n.columns = ['obs']
x = pd.merge(n, m[['STATION', 'CATEGORY']],
             left_on='obs', right_on='STATION',
             how='outer')

x = x[~x['obs'].isna()]
x.to_csv('meta/list_best_inps_match_1.csv', index=False)


""" ___________________  ________________________ """

z = pd.read_csv('meta/NRFA_gdflive_new.csv', dtype=str)
zz = z[~z['STATION'].isna() & z['CATEGORY'].isna()]




m1 = pd.read_excel('meta/EA_gdflive_newnew.xlsx', dtype=str)
m2 = pd.read_csv('meta/NRFA_gdflive_new.csv', dtype=str)

x = pd.merge(m1[['STATION', 'CATEGORY']].dropna(), m2[['STATION', 'CATEGORY']].dropna(),
             on='STATION',
             how='outer')

""" ___________________  ________________________ """

x = pd.read_csv('meta/NRFA_meta.csv', index_col=0, dtype=str)
y = pd.read_excel('meta/EA_gdflive_newnew.xlsx', dtype=str)

z = y.STATION.unique()

m = pd.merge(x, y[['STATION', 'CATEGORY']].dropna(),
             on='STATION',
             how='outer')

""" ___________________  ________________________ """

x = pd.read_csv('meta/metalist_new.csv', dtype=str)
z = pd.merge(y, x, 
         on='STATION',
         how='inner')
z.to_csv('meta/metlist_new_mrg.csv')


""" ___________________  ________________________ """

ids = z.STATION.values

import requests

root = 'https://nrfaapps.ceh.ac.uk/nrfa/ws/time-series?format=json-object&data-type=gdf&station='
root_l = 'https://nrfaapps.ceh.ac.uk/nrfa/ws/time-series?format=json-object&data-type=gdf-live&station='

big = []
for cid in ids:
    try:
        n = requests.get(root+cid).json()
        l = requests.get(root_l+cid).json()
        
        big.append([cid, len(n['data-stream']), len(l['data-stream']), l['station']['easting'], l['station']['northing']])
    except:
        big.append([cid, 0, 0])
q = pd.DataFrame(big)
q.columns = ['STATION', 'gdf', 'gdflive', 'easting', 'northing']

out = pd.merge(z, q,
         on='STATION',
         how='inner').drop_duplicates()

out = out[out['gdf'] > 0]

out.to_csv('meta/meta_new.csv', index=False)

# out = out[out['Data reconciliation status'] == 'Completed']
out[['STATION', 'easting', 'northing']].to_csv('meta/NRFA_meta_gdflive.csv', index=False)


out_f = out[out['gdflive'] > 500]
out_f.to_csv('meta/NRFA_meta_gdfliveonly.csv', index=False)

_out_f = out_f[['STATION', 'easting', 'northing']]
_out_f.columns=['NRFA_ID', 'easting', 'northing']

_out_f.to_csv('meta/NRFA_meta_gdfliveonly_format.csv', index=False)

""" ___________________  ________________________ """



x = pd.read_csv('meta/NRFA_meta_gdfliveonly_format.csv', dtype=str)
y = pd.read_excel('stations_upstream_downstream/nrfa_nearest_sites.xlsx',
                  sheet_name='nrfa_nearest_sites', dtype=str)
z = pd.merge(y, x, 
             left_on='station', right_on='NRFA_ID',
             how='inner')
z.to_csv('meta/NRFA_meta_gdfliveonly_format_updwnstream.csv', index=False)


""" ___________________  ________________________ """



import NRFA_v3 as nrfa

x = nrfa.NRFA(33013)

a = 30e3
x.set_ids_radius(a, a, a)
x.set_ids_updwnstream(a)




""" _____________________________ MORAIN 241  ____________________________ """

# 17.6.

# Get the start and end dates for all of these hourly raingauges
# Maybe find how many are within X km of your gauging station dataset
# See if the period of record is likely to give you enough data to work with

import pandas as pd
import numpy as np

dt0 = pd.read_excel("meta/202003tbr_data_mar2020.xlsx", sheet_name=0)
dt1 = pd.read_excel("meta/202003tbr_data_mar2020.xlsx", sheet_name=1)
dt2 = pd.read_excel("meta/202003tbr_data_mar2020.xlsx", sheet_name=2)


ids = pd.DataFrame(dt0["RAIN_ID"].unique())

lkup = dt1[["ID", "SRC_ID", "EASTING", "NORTHING"]]

x = pd.merge(ids, lkup,
             left_on=ids.columns[0], right_on="ID",
             how="inner").drop(ids.columns[0], axis=1)

dates = []
for st_id in x["ID"]:
    cur = pd.read_csv(f"data/morain_raw/data/{st_id}.csv")
    dates.append([st_id, cur["DAY"].iloc[0], cur["DAY"].iloc[-1], cur.shape[0]])

dates = pd.DataFrame(dates, columns=["ID", "start", "end", "rows"])

y = pd.merge(x, dates,
             on="ID",
             how="inner")

y.set_index("ID", inplace=True)
y.to_csv("meta/MORAIN_241_meta.csv", index=True)


IDS = [33013, 34010, 34012, 34018, 39056, 40017, 46005, 47019, 48001, 49006]
import NRFA_v3 as nrfa

locs = []
for st_id in IDS:
    z = nrfa.NRFA(st_id)
    rows = pd.read_csv(f"data/level1/{st_id}/{st_id}_NRFA.csv")[str(st_id)].shape[0]
    locs.append([st_id, z.east, z.north, rows])
locs = pd.DataFrame(locs, columns=["ID", "east", "north", "rows"])
locs.set_index("ID", inplace=True)

big = pd.DataFrame(np.zeros((len(IDS), y.shape[0]+1)))
big.columns=["ID"]+list(map(str, y.index.values))
big.ID = IDS
big.set_index("ID", inplace=True)

for st_id in IDS:
    e = locs[locs["ID"] == st_id]["east"].values[0]
    n = locs[locs["ID"] == st_id]["north"].values[0]

    for _st_id in y.index:
        _e = y.loc[_st_id, "EASTING"]
        _n = y.loc[_st_id, "NORTHING"]
       
        dist = np.sqrt((e - _e)*(e - _e) + (n - _n)*(n - _n))
        
        big.loc[st_id, str(_st_id)] = dist

big = big.astype(float)
THR = 70000
big[big > THR] = np.nan
big.dropna(thresh=1, axis=1, inplace=True)
# big.to_csv("meta/MORAIN_241_MATT_70km.csv", index=True)


cols = big.columns.astype(int).tolist()
z = y.loc[cols]

bik = big.copy()
for i in bik.index:
    for j in bik.columns:
        if ~np.isnan(bik.loc[i, j]):
            bik.loc[i, j] = y.loc[int(j), "rows"]


for stid in IDS:
    c = pd.DataFrame([big.loc[stid].dropna(), bik.loc[stid].dropna()]).T
    c.columns = ["dist", "rows"]
    c.loc[stid] = [0, locs.loc[stid, "rows"]]
    c.sort_values("dist", inplace=True)
    c.to_csv(f"meta/MORAIN_241/{stid}_70km.csv")

""" ___________________  ________________________ """

