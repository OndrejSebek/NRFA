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



