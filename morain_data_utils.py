import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fuzzywuzzy import fuzz
from fuzzywuzzy import process


''' ______________________ MORAIN data + meta ____________________________ '''

def extract_data():
    """
    Extract data for each station from the combined MORAIN file.

    """
    morain = pd.read_csv('data/morain_raw/raw/morain.csv')
    
    for id_ in morain['ID'].unique():
        cur_mor = morain[morain['ID'] == id_][['DAY', 'PRECIPITATION']]
        cur_mor['DAY'] = pd.to_datetime(cur_mor['DAY'], format='%d-%b-%y')
        cur_mor.loc[cur_mor['DAY'].dt.year >= 2020, 'DAY'] -= pd.DateOffset(years=100)
        cur_mor = cur_mor.sort_values('DAY')
        cur_mor.to_csv('data/morain/data/'+str(id_)+'.csv', index=False)


def merge_data_meta():
    """
    Updated MORAIN meta file with easting, northing.

    """
    morain = pd.read_csv('data/morain/raw/morain.csv')
    meta = pd.read_csv('meta/MORAIN_meta.csv')
    
    ids = pd.DataFrame(morain['ID'].unique())

    upd_meta = pd.merge(ids, meta, left_on=0, right_on='ID')[['ID', 'EASTING', 'NORTHING']]
    upd_meta.to_csv('meta/MORAIN_meta_upd.csv', index=False)


''' ______________________ MO & EA plots (old) ___________________________ '''

def plot_MO_EA_api():
    """
    Plot matching MO & EA gauges ~ API ID.
    
    """
    morain = pd.read_csv('meta/MORAIN_meta_upd.csv')
    morain['ID'] = morain['ID'].astype(str)
    
    # EA_API_lookup = pd.read_csv('meta/COSMOS_meta_updated.csv')[['API_ID', 'NHA_ID']]
    EA = pd.read_csv('meta/EA_API_meta.csv')
    
    # m = pd.merge(EA, EA_API_lookup, left_on='id', right_on='API_ID')
    
    EA['id'] = EA['id'].apply(id_bk_hlpr)
    
    merged = pd.merge(morain, EA, left_on='ID', right_on='id', how='inner')
    
    plt.figure(figsize=(12, 15), dpi=300)
    plt.plot(morain['EASTING'].values, morain['NORTHING'].values,
             linestyle="", marker='x', c='red', label='MO')
    plt.plot(EA['easting'].values, EA['northing'].values,
             linestyle="", c='blue', marker='+', label='EA')
    plt.plot(merged['easting'].values, merged['northing'].values,
             linestyle="", c='green', marker='*', label='ids match')
    plt.legend()
    plt.savefig('meta/EA_MO_station_ids/MORAIN-EA_API___apiID.png')
    plt.close()
    
    
def plot_MO_EA_wiski():
    """
    Plot matching MO & EA gauges ~ WISKI ID.
    
    """
    morain = pd.read_csv('meta/MORAIN_meta_upd.csv')
    morain['ID'] = morain['ID'].astype(str)
    
    id_corr = pd.read_excel('meta/EA_MO_station_ids/Rainfall API ID Lookup_NR Version.xlsx', header=0)
    meta_EA = pd.read_csv('meta/EA_API_meta.csv')
    
    # id format
    meta_EA['id'] = meta_EA['id'].apply(id_bk_hlpr)
    
    merged = pd.merge(meta_EA, id_corr[['ID in API', 'WISKI ID']], 
                      left_on='id', right_on='ID in API',
                      how='inner')
    
    merged_ = pd.merge(merged, morain,
                       left_on='WISKI ID',
                       right_on='ID', how='inner')
    
    plt.figure(figsize=(12, 15), dpi=300)
    plt.plot(morain['EASTING'].values, morain['NORTHING'].values,
             linestyle="", marker='x', c='red', label='MO')
    plt.plot(meta_EA['easting'].values, meta_EA['northing'].values,
             linestyle="", c='blue', marker='+', label='EA')
    plt.plot(merged_['easting'].values, merged_['northing'].values,
             linestyle="", c='green', marker='*', label='ids match')
    plt.legend()
    plt.savefig('meta/EA_MO_station_ids/MORAIN-EA_API___wiskiID.png')
    plt.close()


''' ___________________________ helpers __________________________________ '''

# id format helper
def id_bk_hlpr(x):
    while x[0] == '0':
        x = x[1:]
    return x


''' _________________ EA & MO mapping / matching _________________________ '''

def MO_EA_mapping_dist():
    """
    Map EA api RG meta to MO.
    Selects closest station(/s) for each site and merges meta.

    """
    meta_MO = pd.read_csv('meta/MORAIN_meta.csv', header=0)
    meta_EA = pd.read_csv('meta/EA_API_meta.csv', header=0)
    id_corr = pd.read_excel('meta/EA_MO_station_ids/Rainfall API ID Lookup_NR Version.xlsx', header=0)
    
    mEA = meta_EA #.sort_values('easting', ascending=True)[:100]
    
    big = pd.DataFrame()
    for i in mEA.index:
        # print(mEA.loc[i])
        cur_EA_id = mEA.loc[i, 'id']
        
        # id format (ugh)
        while (cur_EA_id[0] == '0'):
            cur_EA_id = cur_EA_id[1:]
        
        if cur_EA_id in id_corr['ID in API'].astype(str).values:
            # print(cur_EA_id)
            wiski_id = id_corr[id_corr['ID in API'] == str(cur_EA_id)]['WISKI ID'].values
            name = id_corr[id_corr['ID in API'] == str(cur_EA_id)]['Name'].values
            if len(wiski_id) == 0:
                wiski_id = id_corr[id_corr['ID in API'] == int(cur_EA_id)]['WISKI ID'].values
                name = id_corr[id_corr['ID in API'] == int(cur_EA_id)]['Name'].values
        else:
            wiski_id = [cur_EA_id]
            name = ['None']
        
        wiski_id = str(wiski_id[0])
        name = name[0]
        
        c_east = mEA.loc[i, 'easting']
        c_north = mEA.loc[i, 'northing']
            
        meta_MO['dists'] = np.sqrt( (meta_MO['EASTING']-c_east)
                                   *(meta_MO['EASTING']-c_east)
                                   + (meta_MO['NORTHING']-c_north)
                                   *(meta_MO['NORTHING']-c_north) )
        
        # closest = meta_MO.sort_values('dists').iloc[:10]
        # closest['EA_ID'] = [mEA.loc[i, 'id']]*10
        # closest['WISKI_ID'] = [wiski_id]*10 
        
        closest = meta_MO.sort_values('dists').iloc[0]
        closest['EA_ID'] = mEA.loc[i, 'id']
        closest['WISKI_ID'] = wiski_id
        closest['name'] = name
        
        # big[mEA.loc[i, 'id']] = closest.copy()
        
        # closest.to_csv('GS/'+str(mEA.loc[i, 'id'])+'.csv')
        
        if big.empty:
            big = closest.copy()
        else:
            big = pd.concat([big, closest], axis=1)
            
    # big = big.drop_duplicates()
    
    big = big.T
    big = big.reset_index()

    big.to_csv('meta/EA_MO_station_ids/EA_MO_mapping.csv')
    

def MO_EA_match():
    """
    Match MO and EA api RGs based on EA_ID/WISKI_ID/name 

    """
    dt = pd.read_csv('meta/EA_MO_station_ids/EA_MO_mapping.csv',
                     index_col=0)

    dt['EA_match'] = (dt['ID'].astype(str) == dt['EA_ID'].apply(id_bk_hlpr).astype(str))
    dt['WISKI_match'] = (dt['ID'].astype(str) == dt['WISKI_ID'].apply(id_bk_hlpr).astype(str))
    dt['name_match'] = (dt['NAME'].str.upper().str[:3] == dt['name'].str.upper().str[:3])
    
    dt['any_match'] = (dt['EA_match'] | dt['WISKI_match'] | dt['name_match'])
    
    dt.to_csv('meta/EA_MO_station_ids/EA_MO_match.csv')


''' _________________ morain data last datapoint _________________________ '''

def morain_data_years():
    """
    Find date of last data point in MORAIN inp files. Exports plot for all 
    stations. 
    
    Takes a while to pd.read_csv() 14k files.
    
    """
    # adjust to os.getcwd()
    path = 'data/morain_raw/data/'
    ends = []
    
    for file in os.listdir(path):
        ends.append([file, pd.read_csv(path+file, index_col=0).index[-1]])

    ends = pd.DataFrame(ends)
    ends.columns = ['id', 'date']
    
    ends['id'] = ends['id'].apply(lambda x: x[:-4])
    ends['date'] = pd.to_datetime(ends['date'])

    ends.sort_values('date', inplace=True)
    ends.reset_index(inplace=True)
    
    ends.to_csv('meta/morain_data_ends.csv', index=False)
    
    ends[1].plot(figsize=(10, 4))
    plt.savefig('morain_data.png')    

    print(ends[ends[1]=='2019-12-31'].shape[0]/ends.shape[0])


''' _______________ MORAIN - EA API - matching_final _____________________ '''

def MO_EA_map_final():
    meta_MO = pd.read_csv('meta/MORAIN_meta.csv', header=0).astype(str)
    meta_EA_raw = pd.read_csv('meta/EA_API_meta_raw.csv', header=0).astype(str)
    meta_EA = pd.read_csv('meta/EA_API_meta.csv', header=0).astype(str)
    id_corr = pd.read_excel('meta/EA_MO_station_ids/Rainfall API ID Lookup_NR Version.xlsx',
                            header=0).astype(str)
    
    # merge with wiski IDs
    q = pd.merge(meta_EA_raw, id_corr,
                 left_on='id', right_on='ID in API',
                 how='inner').drop('Unnamed: 9', axis=1)
    q.reset_index(inplace=True)
    q.columns = ['idx']+list(q.columns[1:])
    q['matched_on'] = None
    
    # set and format IDs for col to merge on (keeping old format IDs)
    q['ID_in_API_match'] = q['ID in API'].apply(id_bk_hlpr)
    q['WISKI_ID_match'] = q['WISKI ID'].apply(id_bk_hlpr)
    meta_MO['ID_match'] = meta_MO['ID'].apply(id_bk_hlpr)
    meta_MO['SRC_ID_match'] = meta_MO['SRC_ID'].apply(id_bk_hlpr)
    
    meta_MO['NAME'] = meta_MO['NAME'].str.upper()
    q['Name'] = q['Name'].str.upper()

    """ _____________________________ IDS ________________________________ """
    # wiski ~ ID
    q1_ok = pd.merge(q, meta_MO,
                     left_on='WISKI_ID_match', right_on='ID_match',
                     how='inner')
    q1_ok['matched_on'] = 'WISKI-ID'    
    q1_l = q.drop(q1_ok.idx, axis=0)
    
    # API ~ SRC_ID
    q2_ok = pd.merge(q1_l, meta_MO,
                     left_on='ID_in_API_match', right_on='SRC_ID_match',
                     how='inner')
    q2_ok['matched_on'] = 'API-SRC_ID'    
    
    q2_l = q1_l.drop(q2_ok.idx, axis=0)
    
    """ ____________________ DIST + STATION NAME _________________________ """
    n_sub_stations = 5
    q3_ok = []   

    for i in q2_l.index:
        # DIST
        c_east = int(q2_l.loc[i, 'easting'])/1000
        c_north = int(q2_l.loc[i, 'northing'])/1000
    
        dists = np.sqrt( (meta_MO['EASTING'].astype(int)/1000-c_east)**2
                    + (meta_MO['NORTHING'].astype(int)/1000-c_north)**2 )
        dists.sort_values(inplace=True)
        
        sub_stations_dist = dists.loc[dists.index[:n_sub_stations]]
        sub_meta_MO = meta_MO.loc[sub_stations_dist.index]
        
        # NAME (FUZZY)
        names = process.extract(q2_l.loc[i, 'Name'],
                                sub_meta_MO['NAME'].unique(),
                                scorer=fuzz.partial_ratio)
        
        scores = [name[1] for name in names]
        names = [name[0] for name in names]

        # WEIGHTING
        best_name = sub_meta_MO[sub_meta_MO['NAME'] == names[0]]
        best_dist = sub_meta_MO.loc[[sub_meta_MO.index[0]]]
        
        if best_name.shape[0] > 1:
            best_name = best_name.loc[[best_name.index[0]]]
        if best_dist.shape[0] > 1:
            best_dist = best_dist.loc[[best_dist.index[0]]]
    
        if best_name.index == best_dist.index:
            cur = list(q2_l.loc[i].values)
            cur.extend(np.concatenate(sub_meta_MO.loc[best_name.index].values))
            q3_ok.append(cur)
        else:
            # print('\n\n', q2_l.loc[i, 'Name'], '\n', names, '\n', scores, 
            #       '\n', sub_meta_MO['NAME'],'\n', sub_stations_dist,'\n\n')
            pass
    
    # comb    
    q3_ok = pd.DataFrame(q3_ok)
    q3_ok.columns = q2_ok.columns
    q3_ok['matched_on'] = 'DIST+NAME_1' 
    
    q3_l = q2_l.drop(q3_ok.idx, axis=0)
    
    """ ________________________ COMB + EXPORT ___________________________ """    
    
    q_out = pd.concat([q1_ok, q2_ok, q3_ok, q3_l], axis=0,
                      ignore_index=True, sort=False)
    
    q_out.drop('idx', axis=1, inplace=True)
    q_out.sort_values('id')
    q_out = q_out[['id', 'ID in API', 'WISKI ID', 'ID', 'SRC_ID',
                   'Name', 'NAME',
                   'easting', 'northing', 'EASTING', 'NORTHING',
                   'WISKI Easting', 'WISKI Northing',
                   'EASTING in API', 'NORTHING in API', 
                   'WISKI Grid Ref', 'GRIDREF in API',
                   'COUNTRY_CODE', 'HYDROMETRIC_AREA', 'ELEVATION', 'GEOG_PATH',
                   'matched_on', 'ID_match', 'SRC_ID_match',
                   'ID_in_API_match', 'WISKI_ID_match']]
    
    q_out.to_csv('meta/EA_MO_station_ids/final/EA_API_MO_ID_mapping_fin.csv',
                 index=False)
