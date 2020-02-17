import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def extract_data():
    morain = pd.read_csv('data/morain_raw/raw/morain.csv')
    
    for id_ in morain['ID'].unique():
        cur_mor = morain[morain['ID'] == id_][['DAY', 'PRECIPITATION']]
        cur_mor['DAY'] = pd.to_datetime(cur_mor['DAY'], format='%d-%b-%y')
        cur_mor.loc[cur_mor['DAY'].dt.year >= 2020, 'DAY'] -= pd.DateOffset(years=100)
        cur_mor = cur_mor.sort_values('DAY')
        cur_mor.to_csv('data/morain/data/'+str(id_)+'.csv', index=False)


def merge_data_meta():
    morain = pd.read_csv('data/morain/raw/morain.csv')
    meta = pd.read_csv('meta/MORAIN_meta.csv')
    
    ids = pd.DataFrame(morain['ID'].unique())

    upd_meta = pd.merge(ids, meta, left_on=0, right_on='ID')[['ID', 'EASTING', 'NORTHING']]
    upd_meta.to_csv('meta/MORAIN_meta_upd.csv', index=False)






def plot_MO_EA_api():
    morain = pd.read_csv('meta/MORAIN_meta_upd.csv')
    morain['ID'] = morain['ID'].astype(str)
    
    # EA_API_lookup = pd.read_csv('meta/COSMOS_meta_updated.csv')[['API_ID', 'NHA_ID']]
    EA = pd.read_csv('meta/EA_API_meta.csv')
    
    # m = pd.merge(EA, EA_API_lookup, left_on='id', right_on='API_ID')
    
    EA['id'] = EA['id'].apply(id_bk_hlpr)
    
    merged = pd.merge(morain, EA, left_on='ID', right_on='id', how='inner')
    
    
    plt.figure(figsize=(12, 15), dpi=300)
    plt.plot(morain['EASTING'].values, morain['NORTHING'].values, linestyle="", marker='x', c='red', label='MO')
    plt.plot(EA['easting'].values, EA['northing'].values, linestyle="", c='blue', marker='+', label='EA')
    plt.plot(merged['easting'].values, merged['northing'].values, linestyle="", c='green', marker='*', label='ids match')
    plt.legend()
    plt.savefig('meta/EA_MO_station_ids/MORAIN-EA_API___apiID.png')
    plt.close()
    
    
def plot_MO_EA_wiski():
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
    plt.plot(morain['EASTING'].values, morain['NORTHING'].values, linestyle="", marker='x', c='red', label='MO')
    plt.plot(meta_EA['easting'].values, meta_EA['northing'].values, linestyle="", c='blue', marker='+', label='EA')
    plt.plot(merged_['easting'].values, merged_['northing'].values, linestyle="", c='green', marker='*', label='ids match')
    plt.legend()
    plt.savefig('meta/EA_MO_station_ids/MORAIN-EA_API___wiskiID.png')
    plt.close()


# id format helper
def id_bk_hlpr(x):
    while x[0] == '0':
        x = x[1:]
    return x


# map EA api RG meta to MO
#   selects closest station(/s) for each site
#   and merges meta
#
def MO_EA_mapping_dist():
    meta_MO = pd.read_csv('meta/MORAIN_meta.csv', header=0)
    meta_EA = pd.read_csv('meta/EA_API_meta.csv', header=0)
    id_corr = pd.read_excel('meta/EA_MO_station_ids/Rainfall API ID Lookup_NR Version.xlsx', header=0)
    
    mEA = meta_EA #.sort_values('easting', ascending=True)[:100]
    
    big = pd.DataFrame()
    # j=0
    for i in mEA.index:
        # print(mEA.loc[i])
        cur_EA_id = mEA.loc[i, 'id']
        
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
            # j+=1
            # print(j)
        
        wiski_id = str(wiski_id[0])
        name = name[0]
        
        c_east = mEA.loc[i, 'easting']
        c_north = mEA.loc[i, 'northing']
            
        meta_MO['dists'] = np.sqrt( (meta_MO['EASTING']-c_east)*(meta_MO['EASTING']-c_east) + 
                                   (meta_MO['NORTHING']-c_north)*(meta_MO['NORTHING']-c_north) )
        
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
    

# match MO and EA api RGs based on EA_ID/WISKI_ID/name 
#
def MO_EA_match():
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
    path = '../data/morain_raw/data/'
    ends = []
    
    for file in os.listdir(path):
        ends.append([file, pd.read_csv(path+file, index_col=0).index[-1]])

    ends = pd.DataFrame(ends)
    
    ends[0] = ends[0].apply(lambda x: x[:-4])
    ends[1] = pd.to_datetime(ends[1])

    ends.sort_values(1, inplace=True)
    ends.reset_index(inplace=True)

    ends[1].plot(figsize=(10, 4))
    plt.savefig('morain_data.png')    

    print(ends[ends[1]=='2019-12-31'].shape[0]/ends.shape[0])














