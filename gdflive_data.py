import pandas as pd


def fetch_gdflive(st_ids):
    for st_id in st_ids:
        data_nrfa = pd.read_csv(f"https://nrfaapps.ceh.ac.uk/nrfa/ws/time-series/data.csv?format=nrfa-csv&data-type=gdf&station={st_id}", 
                                index_col=0)
        data_live = pd.read_csv(f"https://nrfaapps.ceh.ac.uk/nrfa/ws/time-series/data.csv?format=nrfa-csv&data-type=gdf-live&station={st_id}", 
                                index_col=0)
        
        if data_nrfa.iloc[19][0].isnumeric():
            skiprows = 19
        else:
            skiprows = 20
            
        data_nrfa = data_nrfa.iloc[skiprows:, :1]
        data_nrfa.index = pd.to_datetime(data_nrfa.index)
        
        data_live = data_live.iloc[skiprows:, :1]
        data_live.index = pd.to_datetime(data_live.index)
        
        # print(data_nrfa)
        print(data_live)
        
        data_live.loc[data_nrfa.index[0]:data_nrfa.index[-1]] = data_nrfa.loc[data_nrfa.index[0]:data_nrfa.index[-1]]
        
        data_live.to_csv(f"data/gdf+live/{st_id}.csv")


st_ids = pd.read_csv("data\ens_xgb_imps\list.csv", index_col=None, header=None)

fetch_gdflive(list(st_ids[0]))
# fetch_gdflive([33006])

import NRFA_v3 as nrfa
x = nrfa.NRFA('34010')
x.set_ids_radius(30, 30, 30)
x.fetch_NRFA('gdf-live')
x.fetch_MO()
x.merge_inps('MO', .95)
